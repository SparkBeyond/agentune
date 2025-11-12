from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import attrs
import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection
from frozendict import frozendict
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel

from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.schema import Schema
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.base import (
    BoolFeature,
    CategoricalFeature,
    Feature,
    FloatFeature,
    IntFeature,
    LlmFeature,
)
from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import (
    DataFormatter,
)
from agentune.analyze.feature.gen.insightful_text_generator.prompts import (
    QUERY_FEATURE_PROMPT,
    get_output_instructions,
)
from agentune.analyze.feature.gen.insightful_text_generator.schema import (
    PARSER_OUT_FIELD,
    BoolResponse,
    FloatResponse,
    IntResponse,
    Query,
    StrResponse,
)
from agentune.analyze.join.base import JoinStrategy
from agentune.analyze.util.attrutil import frozendict_converter


@frozen
class InsightfulTextFeature[T](LlmFeature[T]):

    model: LLMWithSpec
    formatter: DataFormatter
    formatter_param_name: str
    prompt_template: str
    prompt_parameters: frozendict[str, str] = attrs.field(converter=frozendict_converter)

    # Implement the abstract properties
    @property
    def params(self) -> Schema:
        return self.formatter.params
    
    @property
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        return self.formatter.secondary_tables

    @property
    def join_strategies(self) -> Sequence[JoinStrategy]:
        return self.formatter.join_strategies
    
    @property
    @abstractmethod
    def parser_model(self) -> type[BaseModel]:
        """The Pydantic model used to parse LLM responses."""

    async def acompute(self, args: tuple[Any, ...],
                       conn: DuckDBPyConnection) -> T | None:
        # convert to dataframe
        df = pl.DataFrame(
            {col.name: [value] for col, value in zip(self.params.cols, args, strict=True)},
            schema=self.params.to_polars()
        )
        # format the data
        formatted_value = (await self.formatter.aformat_batch(Dataset(self.params, df), conn))[0]
        params = dict(self.prompt_parameters.copy())
        params[self.formatter_param_name] = formatted_value

        # create the prompt
        prompt = self.prompt_template.format(**params)

        # call LLM
        sllm = self.model.llm.as_structured_llm(self.parser_model)
        messages = [ChatMessage(role='user', content=prompt)]
        response = await sllm.achat(messages)
        # parse the response
        out = response.raw.model_dump()[PARSER_OUT_FIELD]

        return out


# Dtype-specific feature classes
@frozen
class InsightfulBoolFeature(InsightfulTextFeature[bool], BoolFeature):
    """Boolean insightful text feature."""

    # Redeclare with concrete types to work around attrs issue
    default_for_missing: bool

    @property
    def parser_model(self) -> type[BaseModel]:
        return BoolResponse


@frozen
class InsightfulIntFeature(InsightfulTextFeature[int], IntFeature):
    """Integer insightful text feature."""

    # Redeclare with concrete types to work around attrs issue
    default_for_missing: int

    @property
    def parser_model(self) -> type[BaseModel]:
        return IntResponse


@frozen
class InsightfulFloatFeature(InsightfulTextFeature[float], FloatFeature):
    """Float insightful text feature."""

    # Redeclare with concrete types to work around attrs issue
    default_for_missing: float

    @property
    def parser_model(self) -> type[BaseModel]:
        return FloatResponse


@frozen
class InsightfulCategoricalFeature(InsightfulTextFeature[str], CategoricalFeature):
    """Categorical insightful text feature."""

    # Redeclare with concrete types to work around attrs issue
    default_for_missing: str

    @property
    def parser_model(self) -> type[BaseModel]:
        return StrResponse


def create_feature(query: Query, formatter: DataFormatter, model: LLMWithSpec) -> Feature:
    """Create an insightful text feature based on the query definition."""
    dtype_to_class: dict[types.Dtype, type] = {
        types.boolean: InsightfulBoolFeature,
        types.int32: InsightfulIntFeature,
        types.float64: InsightfulFloatFeature
    }
    feature_class = dtype_to_class.get(query.return_type, InsightfulCategoricalFeature)
    output_instructions = get_output_instructions(query.return_type)

    prompt_parameters = {
        'instance_description': formatter.description,
        'query_text': query.query_text,
        'output_instructions': output_instructions
    }
    
    # Create technical description based on the feature type and query
    technical_description = (
        f'Input: {formatter.description}\n'
        f'Query: "{query.query_text}"\n'
        f'Output Type: {query.return_type}\n'
        f'Uses LLM: {model.spec.model_name}'
    )
    
    # Common parameters for all feature types
    common_kwargs = {
        'name': query.name,
        'description': query.query_text,
        'technical_description': technical_description,
        'formatter': formatter,
        'formatter_param_name': 'instance',
        'prompt_template': QUERY_FEATURE_PROMPT,
        'prompt_parameters': prompt_parameters,
        'model': model,
    }
    
    # Type-specific parameters
    # Note: The defaults defined here are meant to be naive placeholders
    if feature_class == InsightfulBoolFeature:
        common_kwargs['default_for_missing'] = False
    elif feature_class == InsightfulIntFeature:
        common_kwargs['default_for_missing'] = 0
    elif feature_class == InsightfulFloatFeature:
        common_kwargs.update({
            'default_for_missing': 0.0,
            'default_for_nan': 0.0,
            'default_for_infinity': 0.0,
            'default_for_neg_infinity': 0.0,
        })
    elif feature_class == InsightfulCategoricalFeature:
        if not isinstance(query.return_type, types.EnumDtype):
            raise TypeError('Categorical features must have EnumDtype return type')
        common_kwargs.update({
            'categories': query.return_type.values,
            'default_for_missing': CategoricalFeature.other_category,
        })

    return feature_class(**common_kwargs)
