"""Type detection utilities for analyzing data and determining appropriate data types."""

import logging

import polars as pl
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field

from agentune.analyze.core import types
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.gen.insightful_text_generator.prompts import (
    CATEGORICAL_OPTIMIZER_PROMPT,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import Query

logger = logging.getLogger(__name__)


def detect_bool_type(data: pl.Series, max_error_percentage: float) -> types.Dtype | None:
    """Detect if the data represents boolean values and return the boolean type.
    
    Args:
        data: Polars Series with the data (should be non-null values)
        max_error_percentage: Maximum allowed percentage of values that can fail to match boolean-like values (e.g., 0.01 = 1%)

    Returns:
        types.boolean if data is detected as boolean-like, None otherwise
    """
    # Check if all values are boolean-like
    lower_values = data.str.to_lowercase()
    bool_values = {'true', 'false', 't', 'f', 'yes', 'no', 'y', 'n', '1', '0'}
    bool_count = lower_values.is_in(bool_values).sum()
    if bool_count >= len(data) * (1 - max_error_percentage):
        return types.boolean    
    return None


def detect_int_type(data: pl.Series, max_error_percentage: float) -> types.Dtype | None:
    """Detect if the data represents integer values and return the integer type.
    
    Args:
        data: Polars Series with the data (should be non-null values)
        max_error_percentage: Maximum allowed percentage of values that can fail to cast to int (e.g., 0.05 = 5%)
        
    Returns:
        types.int32 if data is detected as integer, None otherwise
    """
    int_series = data.cast(pl.Int32, strict=False)
    # count null values after casting
    n_nulls = int_series.null_count()
    if n_nulls <= len(data) * max_error_percentage:
        return types.int32    
    return None


def detect_float_type(data: pl.Series, max_error_percentage: float) -> types.Dtype | None:
    """Detect if the data represents float values and return the float type.
    
    Args:
        data: Polars Series with the data (should be non-null values)
        max_error_percentage: Maximum allowed percentage of values that can fail to cast to float (e.g., 0.05 = 5%)

    Returns:
        types.float64 if data is detected as float, None otherwise
    """
    float_series = data.cast(pl.Float64, strict=False)
    # count null values after casting
    n_nulls = float_series.null_count()
    if n_nulls <= len(data) * max_error_percentage:
        return types.float64
    return None


def detect_categorical_type(data: pl.Series, max_categorical: int, min_threshold_percentage: float, 
                            min_coverage: float, max_chars: int) -> types.Dtype | None:
    """Detect if the data can be represented as categorical/enum values and return the enum type.
    
    Args:
        data: Polars Series with the data (should be non-null values)
        max_categorical: Maximum number of categories for enum
        min_threshold_percentage: Minimum percentage of total data a category must represent
        min_coverage: Minimum coverage the top categories must provide
        max_chars: Maximum number of characters for each category value

    Returns:
        types.EnumDtype if data is detected as categorical, None otherwise
    """
    total_count = len(data)
    min_count_threshold = int(total_count * min_threshold_percentage)
    
    # Get value counts
    value_counts = data.value_counts().sort('count', descending=True)
    
    # Filter values that meet the minimum percentage threshold
    qualifying_values = value_counts.filter(pl.col('count') >= min_count_threshold)
    
    if len(qualifying_values) == 0:
        return None

    # Take top max_categorical values
    top_values = qualifying_values.head(max_categorical)
    column_name = top_values.columns[0]  # Get the actual column name

    # If empty string is one of the categories, return None
    if top_values.filter(pl.col(column_name) == '').height > 0:
        return None

    # Check if any value exceeds max_chars
    if top_values.select(pl.col(column_name).str.len_chars().max()).item() > max_chars:
        return None

    # Calculate coverage of top categories
    top_coverage = top_values.select('count').sum().item() / total_count
    
    # Check if coverage meets minimum requirement
    if top_coverage >= min_coverage:
        # The value_counts returns columns with original column name and "count"
        categories = [str(row[0]) for row in top_values.select(column_name).iter_rows()]
        
        return types.EnumDtype(*categories)
    
    return None


def decide_dtype(query: Query, data: pl.Series, max_categorical: int, max_error_percentage: float = 0.05,
                 min_threshold_percentage: float = 0.01, min_categorical_coverage: float = 0.9, max_categorical_chars: int = 50) -> types.Dtype:
    """Decide the appropriate data type for a query based on the data values.
    
    Args:
        query: The query object
        data: Polars Series with the data
        max_categorical: Maximum number of categories for enum (default: 10)
        min_threshold_percentage: Minimum percentage of total data a category must represent (default: 0.05 = 5%)
        min_categorical_coverage: Minimum coverage the top categories must provide (default: 0.9 = 90%)
        max_categorical_chars: Maximum number of characters for each category value (default: 50)

    Returns:
        The appropriate Dtype for the data
    """
    # Drop null values before processing
    data = data.drop_nulls()
    
    if data.is_empty():
        logger.warning(f'Empty data for query {query.name}, defaulting to string')
        return types.string
    
    # Get unique values and counts for logging
    n_unique = len(data.unique())
    
    # Try each type detector in order
    dtype = detect_bool_type(data, max_error_percentage)
    if dtype is not None:
        logger.debug(f'Query {query.name}: {n_unique} unique values out of {len(data)} total -> {dtype}')
        return dtype

    dtype = detect_int_type(data, max_error_percentage)
    if dtype is not None:
        logger.debug(f'Query {query.name}: {n_unique} unique values out of {len(data)} total -> {dtype}')
        return dtype

    dtype = detect_float_type(data, max_error_percentage)
    if dtype is not None:
        logger.debug(f'Query {query.name}: {n_unique} unique values out of {len(data)} total -> {dtype}')
        return dtype

    dtype = detect_categorical_type(data, max_categorical, min_threshold_percentage, min_categorical_coverage, max_categorical_chars)
    if dtype is not None:
        # Log histogram in single line for categorical
        value_counts = data.value_counts().sort('count', descending=True)
        histogram_str = ', '.join([f"'{row[0]}': {row[1]}" for row in value_counts.iter_rows()])
        logger.debug(f'Query {query.name}: {n_unique} unique values out of {len(data)} total -> {dtype} | Histogram: {histogram_str}')
        return dtype

    # Default to string
    logger.debug(f'Query {query.name}: {n_unique} unique values out of {len(data)} total -> string (insufficient coverage)')
    return types.string


class CategoricalOptimizerResponse(BaseModel):
    query_name: str = Field(..., description='Clear feature name')
    categories: list[str] = Field(..., description='List of category names')
    query_text: str = Field(..., description='Refined query that maps to your categories')


async def cast_to_categorical(query: Query, data: pl.Series, max_categorical: int, llm: LLMWithSpec) -> Query:
    """Cast the data to a categorical type and return the updated query with the new enum dtype.

    Args:
        query: The query object
        data: Polars Series with the data (should be non-null values)
        max_categorical: Maximum number of categories for enum
        llm: The language model

    Returns:
        Query: The updated query with the new enum dtype
    """
    # Get historical answers histogram for the prompt
    value_counts = data.value_counts().sort('count', descending=True)
    # Create histogram string: "answer1": count1, "answer2": count2, ...
    histogram_entries = [f'"{row[0]}": {row[1]}' for row in value_counts.iter_rows()]
    answers_hist = '{' + ', '.join(histogram_entries) + '}'
    
    # Create prompt parameters
    prompt_params = {
        'query_name': query.name,
        'query_text': query.query_text,
        'max_categorical': max_categorical,
        'answers_hist': answers_hist
    }
    
    # Format the prompt
    prompt = CATEGORICAL_OPTIMIZER_PROMPT.format(**prompt_params)
    
    # Create structured LLM
    sllm = llm.llm.as_structured_llm(CategoricalOptimizerResponse)
    
    # Call LLM
    messages = [ChatMessage(role='user', content=prompt)]
    response = await sllm.achat(messages)
    response_obj: CategoricalOptimizerResponse = response.raw

    # remove empty string category if exists
    if '' in response_obj.categories:
        response_obj.categories.remove('')

    # Check response
    if not response_obj.query_name:
        raise ValueError('query_name cannot be empty')
    if not response_obj.categories:
        raise ValueError('categories cannot be empty')
    if not response_obj.query_text:
        raise ValueError('query_text cannot be empty')
    if len(response_obj.categories) > max_categorical:
        raise ValueError(f'Too many categories: {len(response_obj.categories)} > {max_categorical}')

    categories = response_obj.categories

    # Create updated query
    updated_query = Query(
        name=response_obj.query_name,
        query_text=response_obj.query_text,
        return_type=types.EnumDtype(*categories)
    )
    
    return updated_query
    
