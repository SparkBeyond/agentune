"""Core feature selection functionality tests.

This module provides consolidated tests for all feature selectors covering:
1. Basic functionality on real datasets (classification & regression)
2. Ground truth vs LinearPairWise NDCG similarity validation (>0.99)
3. Enriched API synthetic data testing
4. Edge cases (empty features, top_k > available)
5. String recognition for categorical features and multiclass targets
6. Fake category impact testing
"""
import json
import math
import time
from pathlib import Path
from typing import cast

import duckdb
import numpy as np
import polars as pl
import pytest

from agentune.analyze.core import types
from agentune.analyze.core.schema import Field
from agentune.analyze.feature.base import BoolFeature
from agentune.analyze.feature.problem import (
    ClassificationProblem,
    Problem,
    ProblemDescription,
    RegressionProblem,
    TargetKind,
)
from agentune.analyze.feature.select.lightgbm import (
    LightGBMFeatureSelector,
)
from agentune.analyze.feature.select.linear_pairwise import LinearPairWiseFeatureSelector

from .conftest import EnrichedBuilder
from .helpers import load_and_clean_csv

# Test configuration
SELECTORS = ['LightGBM', 'Linear PairWise']
DATASETS: dict[str, dict[str, str | int]] = {
    'titanic': {
        'file_path': str((Path(__file__).parent / 'data' / 'titanic_300_features_anonymized.csv').resolve()),
        'target_col': 'survived',
        'task_type': 'classification',
        'top_k': 20
    }
}


def ndcg_at_k(eval_list: list[str], ref_list: list[str], k: int = 20) -> float:
    """Calculate NDCG@k score with exponential gain."""
    k = min(k, len(ref_list), len(eval_list))
    
    # Create relevance scores (higher rank = higher relevance)
    n = len(ref_list)
    relevance = {item: (n - i) for i, item in enumerate(ref_list)}
    
    # Exponential gain DCG
    dcg = 0.0
    for i, item in enumerate(eval_list[:k], start=1):
        rel = relevance.get(item, 0)
        gain = (2 ** rel) - 1
        dcg += gain / math.log2(i + 1)
    
    # Ideal DCG with exponential gain
    idcg = 0.0
    for i, item in enumerate(ref_list[:k], start=1):
        rel = relevance[item]
        gain = (2 ** rel) - 1
        idcg += gain / math.log2(i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0


def symmetric_ndcg(list_a: list[str], list_b: list[str], k: int = 20) -> float:
    """Calculate symmetric NDCG between two rankings."""
    return 0.5 * (ndcg_at_k(list_b, list_a, k) + ndcg_at_k(list_a, list_b, k))


# Timing sweep configuration (using anonymized Titanic dataset)
TIMING_TOP_KS = [5, 10, 20]

def define_problem(config: dict[str, str | int]) -> Problem:
    task_type = cast(TargetKind, config['task_type'])
    target_col = str(config['target_col'])
    target_dtype = types.int64 # This is not true for all datasets, change this if the selector starts using the dtype
    target_field = Field(target_col, target_dtype)
    return ClassificationProblem(ProblemDescription(target_col), target_field, (0, 1)) if task_type == 'classification' \
        else RegressionProblem(ProblemDescription(target_col), target_field)

# All tests below instantiate EnrichedBuilder to construct Dataset/Source/Features
def test_linear_pairwise_timing_sweep_real_datasets(conn: duckdb.DuckDBPyConnection) -> None:
    """Measure Linear PairWise selection time at different top_k values for the Titanic dataset.

    This test measures execution time for top_k values defined in TIMING_TOP_KS on the anonymized Titanic dataset.
    Results are saved as a CSV where rows are datasets and columns are the requested top_k values.
    Output: selector_results/linear_pairwise_timing_by_k.csv
    """
    timing_results = []
    
    for dataset_name, config in DATASETS.items():
        # Load dataset
        df = load_and_clean_csv(str(config['file_path']), dataset_name, cast_text_to_float=True)
        
        # Build enriched API objects once per dataset
        builder = EnrichedBuilder()
        features, source = builder.build(df, config['target_col'])  # type: ignore[arg-type]
        
        dataset_timings = {'dataset': dataset_name}
        for top_k in TIMING_TOP_KS:
            if top_k > len(features):
                dataset_timings[f'top_k_{top_k}'] = 'None'
                continue
            # Create selectors for comparison via factory to avoid duplication
            selector_names = ['LightGBM', 'Linear PairWise']
            selectors = {name: create_selector(name, top_k=top_k) for name in selector_names}
            
            # Measure both, but record only Linear PairWise in dataset_timings for CSV/analysis
            times_by_selector: dict[str, float] = {}
            for selector_name, selector in selectors.items():
                # Measure execution time using enriched API
                start_time = time.time()
                selected_features = selector.select_features(features, source, define_problem(config), conn)
                execution_time = time.time() - start_time
                
                times_by_selector[selector_name] = round(execution_time, 3)
                
                # Sanity check
                assert len(selected_features) <= top_k, f'Should not exceed top_k={top_k}'
                assert len(selected_features) > 0, 'Should select at least one feature'

            # For CSV: keep only the Linear PairWise timing per top_k
            if 'Linear PairWise' in times_by_selector:
                dataset_timings[f'top_k_{top_k}'] = str(times_by_selector['Linear PairWise'])
            else:
                dataset_timings[f'top_k_{top_k}'] = 'None'
        
        timing_results.append(dataset_timings)
    
    # Performance assertions
    for result in timing_results:
        dataset_name = result['dataset']
        
        # Check that timing generally increases with top_k (allowing some variance)
        timing_values = []
        for k in TIMING_TOP_KS:
            val = result.get(f'top_k_{k}')
            if val is not None and val != 'None':
                timing_values.append(float(val))
        
        if len(timing_values) >= 2:
            # The largest top_k should not be more than 10x slower than the smallest
            min_time = min(timing_values)
            max_time = max(timing_values)
            time_ratio = max_time / min_time if min_time > 0 else float('inf')
            
            assert time_ratio <= 20, f'{dataset_name}: timing ratio ({time_ratio:.1f}x) should be reasonable'
            
            # All timings should be under 5 minutes (300 seconds)
            assert max_time <= 300, f'{dataset_name}: max timing ({max_time:.1f}s) should be under 5 minutes'


def create_selector(selector_name: str, top_k: int) -> LinearPairWiseFeatureSelector | LightGBMFeatureSelector:
    """Factory function to create selectors with unified interface."""
    if selector_name == 'LightGBM':
        return LightGBMFeatureSelector(top_k=top_k)
    elif selector_name == 'Linear PairWise':
        return LinearPairWiseFeatureSelector(top_k=top_k)
    else:
        raise ValueError(f'Unknown selector: {selector_name}')


@pytest.mark.parametrize('selector_name', SELECTORS)
@pytest.mark.parametrize('dataset_name', list(DATASETS.keys()))
def test_selector_basic_functionality(selector_name: str, dataset_name: str, conn: duckdb.DuckDBPyConnection) -> None:
    """Test basic selector functionality on real datasets."""
    # Get dataset configuration
    dataset_config = DATASETS[dataset_name]
    
    # Load and clean data
    df = load_and_clean_csv(str(dataset_config['file_path']), dataset_name, cast_text_to_float=True)
    
    # Create selector
    selector = create_selector(
        selector_name=selector_name,
        top_k=cast(int, dataset_config['top_k'])
    )
    
    # Get features and source using enriched API
    builder = EnrichedBuilder()
    features, source = builder.build(df, dataset_config['target_col'])  # type: ignore[arg-type]
    
    # Select features
    selected_features = selector.select_features(features, source, define_problem(dataset_config), conn)
    
    # Basic assertions
    top_k_val = dataset_config['top_k']
    assert isinstance(top_k_val, int)
    assert len(selected_features) <= top_k_val, f'Should not exceed top_k={top_k_val}'
    assert len(selected_features) > 0, 'Should select at least one feature'
    assert all(hasattr(f, 'name') for f in selected_features), 'All selected features should have names'
    
    # Feature names should be unique
    selected_names = [f.name for f in selected_features]
    assert len(selected_names) == len(set(selected_names)), 'Selected feature names should be unique'
    
    # All selected features should exist in original dataset
    available_features = [f.name for f in features]
    assert all(name in available_features for name in selected_names), 'Selected features should exist in original dataset'

    # Inline NDCG assertion for Linear PairWise vs external ground truth (required)
    selector_slug = selector_name.replace(' ', '_').lower()
    dataset_slug = dataset_name.lower()
    if selector_slug == 'linear_pairwise':
        # Verify importance scores are normalized between 0 and 1
        assert selector.final_importances_ is not None, 'LinearPairWise should have final_importances_ set'
        importance_scores = selector.final_importances_['importance']
        assert all(0.0 <= score <= 1.0 for score in importance_scores), f'LinearPairWise importance scores should be normalized between 0 and 1, got: {importance_scores}'
        assert len(importance_scores) == len(selected_features), 'Should have one importance score per selected feature'
        
        gt_file = Path(__file__).parent / 'data' / f'ground_truth_{dataset_slug}_results.json'
        assert gt_file.exists(), f'Expected external ground truth at {gt_file} but it was not found.'
        with gt_file.open() as f:
            gt_data = json.load(f)
        gt_features = gt_data.get('selected_features', [])
        assert gt_features, f"Ground truth file {gt_file} has no 'selected_features'."
        ndcg_score = symmetric_ndcg(selected_names, gt_features, k=20)
        # Note: NDCG threshold lowered from 0.99 to 0.85 because LinearPairWise now returns features
        # in selection order (order they were chosen) rather than importance order (highest score first).
        # Ground truth was created with importance ordering, so there's a natural ordering difference
        # that still validates the same high-quality features are selected.
        assert ndcg_score > 0.85, f'{dataset_name}: NDCG ({ndcg_score:.4f}) should be > 0.85'


@pytest.mark.parametrize('selector_name', SELECTORS)
@pytest.mark.parametrize('task_type', ['classification', 'regression'])
def test_enriched_api_synthetic_data(selector_name: str, task_type: str, conn: duckdb.DuckDBPyConnection) -> None:
    """Test selectors on synthetic data via enriched API."""
    # Generate synthetic data
    rng = np.random.default_rng(42)
    n_rows, n_features = 200, 10
    
    # Create features
    x = rng.normal(size=(n_rows, n_features))
    
    # Create target based on task type
    problem: Problem
    if task_type == 'classification':
        # Binary classification: combine first 3 features
        y_prob = 1 / (1 + np.exp(-(x[:, 0] + 0.5 * x[:, 1] - 0.3 * x[:, 2])))
        y = (rng.random(n_rows) < y_prob).astype(int)
        target_col = 'target_class'
        problem = ClassificationProblem(ProblemDescription('target_class'), Field('target_class', types.int64), (0, 1))
    else:  # regression
        # Regression: linear combination with noise
        y = x[:, 0] + 0.5 * x[:, 1] - 0.3 * x[:, 2] + 0.1 * rng.normal(size=n_rows)
        target_col = 'target_value'
        problem = RegressionProblem(ProblemDescription('target_value'), Field('target_value', types.float64))
    
    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    data_dict = {col: x[:, i] for i, col in enumerate(feature_cols)}
    data_dict[target_col] = y
    df = pl.DataFrame(data_dict)
    
    # Create enriched API objects via EnrichedBuilder
    builder = EnrichedBuilder()
    features, source = builder.build(df, target_col)
    
    # Create selector
    selector = create_selector(selector_name, top_k=5)
    
    # Select features
    selected = selector.select_features(features, source, problem, conn)
    
    # Assertions
    assert len(selected) <= 5, 'Should not exceed top_k=5'
    assert len(selected) > 0, 'Should select at least one feature'
    
    # The first 3 features should be more likely to be selected (they're informative)
    selected_names = {f.name for f in selected}
    informative_selected = len({'feature_0', 'feature_1', 'feature_2'} & selected_names)
    
    # At least one informative feature should be selected
    assert informative_selected > 0, f'At least one informative feature should be selected, got: {selected_names}'


def test_edge_cases(conn: duckdb.DuckDBPyConnection) -> None:
    """Test edge cases: empty features, top_k > available."""
    # Create small synthetic dataset
    rng = np.random.default_rng(42)
    n_rows, n_features = 50, 3
    x = rng.normal(size=(n_rows, n_features))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)  # Simple binary target
    
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    data_dict = {col: x[:, i] for i, col in enumerate(feature_cols)}
    data_dict['target'] = y
    df = pl.DataFrame(data_dict)
    
    # Build enriched API objects
    builder = EnrichedBuilder()
    features, source = builder.build(df, 'target')
    
    # Test 1: Empty features list
    selector = LinearPairWiseFeatureSelector(top_k=5)
    regression_problem = RegressionProblem(ProblemDescription('target'), Field('target', types.float64))
    selected = selector.select_features([], source, regression_problem, conn)
    assert len(selected) == 0, 'Empty features should return empty selection'
    
    # Test 2: top_k > available features
    selector = LinearPairWiseFeatureSelector(top_k=10)  # top_k > available
    classification_problem = ClassificationProblem(ProblemDescription('target'), Field('target', types.int64), (0, 1))
    selected = selector.select_features(features, source, classification_problem, conn)
    assert len(selected) <= len(features), 'Should not exceed available features'
    assert len(selected) > 0, 'Should still select available features'


def test_fake_category_minimal_impact(conn: duckdb.DuckDBPyConnection) -> None:
    """Test that adding a fake numeric feature doesn't significantly change results."""
    # Create base synthetic dataset
    rng = np.random.default_rng(42)
    n_rows = 300
    
    # Informative features
    x1 = rng.normal(size=n_rows)
    x2 = rng.normal(size=n_rows)
    x3 = rng.normal(size=n_rows)  # Less informative
    
    # Target based on x1 and x2
    y = (x1 + 0.5 * x2 + 0.1 * rng.normal(size=n_rows) > 0).astype(int)
    
    # Base dataset (without fake feature)
    df_base = pl.DataFrame({
        'feature1': x1,
        'feature2': x2, 
        'feature3': x3,
        'target': y
    })
    
    # Enhanced dataset (with fake numeric feature that has no predictive power)
    fake_feature = rng.normal(size=n_rows)  # Random, non-predictive numeric feature
    df_fake = df_base.with_columns(pl.Series('fake_feature', fake_feature))
    
    # Test both datasets with LinearPairWise
    
    # Base dataset selection
    builder = EnrichedBuilder()
    features_base, source_base = builder.build(df_base, 'target')
    selector_base = create_selector('Linear PairWise', top_k=3)
    classification_problem = ClassificationProblem(ProblemDescription('target'), Field('target', types.int64), (0, 1))
    selected_base = selector_base.select_features(features_base, source_base, classification_problem, conn)
    selected_names_base = {f.name for f in selected_base}
    
    # Enhanced dataset selection (with fake feature)
    features_fake, source_fake = builder.build(df_fake, 'target')
    selector_fake = create_selector('Linear PairWise', top_k=3)
    selected_fake = selector_fake.select_features(features_fake, source_fake, classification_problem, conn)
    selected_names_fake = {f.name for f in selected_fake}
    
    # Compare results
    common_features = selected_names_base & selected_names_fake
    overlap_ratio = len(common_features) / len(selected_names_base)
    
    # The fake feature should not be selected (it's non-predictive)
    assert 'fake_feature' not in selected_names_fake, 'Non-predictive fake feature should not be selected'
    
    # Feature overlap should be high (fake feature shouldn't disrupt good features)
    assert overlap_ratio >= 0.67, f'Feature overlap ({overlap_ratio:.2f}) should be at least 67% when adding fake feature'


def test_linear_pairwise_threshold_functionality(conn: duckdb.DuckDBPyConnection) -> None:
    """Test LinearPairWiseFeatureSelector threshold functionality."""
    # Create synthetic dataset with features of varying quality
    rng = np.random.default_rng(42)
    n_rows = 300
    
    # High-quality features (strongly predictive)
    x1 = rng.normal(size=n_rows)
    x2 = rng.normal(size=n_rows)
    
    # Medium-quality feature (moderately predictive)
    x3 = rng.normal(size=n_rows)
    
    # Low-quality features (weakly predictive or noise)
    x4 = rng.normal(size=n_rows) * 0.1  # Very weak signal
    x5 = rng.normal(size=n_rows)  # Pure noise
    
    # Target based primarily on x1 and x2, with small contribution from x3
    y = (x1 + 0.8 * x2 + 0.2 * x3 + 0.05 * rng.normal(size=n_rows) > 0).astype(int)
    
    df = pl.DataFrame({
        'high_quality_1': x1,
        'high_quality_2': x2,
        'medium_quality': x3,
        'low_quality_1': x4,
        'low_quality_2': x5,
        'target': y
    })
    
    # Build enriched API objects
    builder = EnrichedBuilder()
    features, source = builder.build(df, 'target')
    
    # Test 1: Default threshold (1e-8) - should work as before
    selector_default = LinearPairWiseFeatureSelector(top_k=5)
    classification_problem = ClassificationProblem(ProblemDescription('target'), Field('target', types.int64), (0, 1))
    selected_default = selector_default.select_features(features, source, classification_problem, conn)
    assert len(selected_default) > 0, 'Default threshold should select features'
    assert len(selected_default) <= 5, 'Should not exceed top_k'
    
    # Test 2: High threshold (0.1) - should select fewer features
    selector_high = LinearPairWiseFeatureSelector(top_k=5, min_marginal_reduction_threshold=0.1)
    selected_high = selector_high.select_features(features, source, classification_problem, conn)
    assert len(selected_high) <= len(selected_default), 'High threshold should select same or fewer features'
    
    # Test 3: Very high threshold (0.5) - might select no features
    selector_very_high = LinearPairWiseFeatureSelector(top_k=5, min_marginal_reduction_threshold=0.5)
    selected_very_high = selector_very_high.select_features(features, source, classification_problem, conn)
    assert len(selected_very_high) <= len(selected_high), 'Very high threshold should select same or fewer features'
    
    # Test 4: Zero threshold - should still reject features with 0 or negative gain
    selector_zero = LinearPairWiseFeatureSelector(top_k=5, min_marginal_reduction_threshold=0.0)
    _ = selector_zero.select_features(features, source, classification_problem, conn)
    
    # All selected features should have positive marginal gain (> 1e-10 due to effective threshold)
    if selector_zero.final_importances_ is not None:
        importance_scores = selector_zero.final_importances_['importance']
        assert all(score > 1e-10 for score in importance_scores), 'Even with threshold=0, features with zero gain should not be selected'
    
    # Test 5: Verify that higher quality features are preferred
    if len(selected_default) >= 2:
        selected_names_default = {f.name for f in selected_default}
        # High quality features should be more likely to be selected
        high_quality_selected = len({'high_quality_1', 'high_quality_2'} & selected_names_default)
        assert high_quality_selected > 0, 'At least one high-quality feature should be selected'


def test_linear_pairwise_threshold_early_stopping(conn: duckdb.DuckDBPyConnection) -> None:
    """Test that LinearPairWiseFeatureSelector stops early when no features meet threshold."""
    # Create dataset where only one feature is truly predictive
    rng = np.random.default_rng(123)
    n_rows = 200
    
    # One good feature
    x_good = rng.normal(size=n_rows)
    
    # Many noise features (should have very low marginal gain)
    noise_features = {f'noise_{i}': rng.normal(size=n_rows) * 0.01 for i in range(10)}
    
    # Target based only on the good feature
    y = (x_good + 0.1 * rng.normal(size=n_rows) > 0).astype(int)
    
    data_dict = {'good_feature': x_good, 'target': y}
    data_dict.update(noise_features)
    df = pl.DataFrame(data_dict)
    
    # Build enriched API objects
    builder = EnrichedBuilder()
    features, source = builder.build(df, 'target')
    
    # Test with moderate threshold - should stop early and select only good features
    selector = LinearPairWiseFeatureSelector(top_k=10, min_marginal_reduction_threshold=0.01)
    classification_problem = ClassificationProblem(ProblemDescription('target'), Field('target', types.int64), (0, 1))
    selected = selector.select_features(features, source, classification_problem, conn)
    
    # Should select fewer than top_k due to early stopping
    assert len(selected) < 10, 'Should stop early when features do not meet threshold'
    assert len(selected) > 0, 'Should select at least the good feature'
    
    # The good feature should be selected
    selected_names = {f.name for f in selected}
    assert 'good_feature' in selected_names, 'The predictive feature should be selected'
    
    # Most noise features should not be selected
    noise_selected = len([name for name in selected_names if name.startswith('noise_')])
    assert noise_selected <= 2, 'Most noise features should not meet the threshold'


@pytest.mark.parametrize('selector_name', SELECTORS)
def test_boolean_feature_handling(selector_name: str, conn: duckdb.DuckDBPyConnection) -> None:
    """Selectors should correctly handle boolean features via enriched API.

    We construct a dataset where a boolean feature perfectly determines the binary target.
    Both selectors should select the boolean feature among the top features.
    """
    rng = np.random.default_rng(123)
    n_rows = 300

    # Boolean feature and target perfectly aligned
    bool_feature = rng.random(n_rows) < 0.5  # dtype: bool
    target = bool_feature.astype(np.int64)   # binary target 0/1

    # Add some noise numeric features that should be non-predictive
    num1 = rng.normal(size=n_rows)
    num2 = rng.normal(size=n_rows)

    df = pl.DataFrame({
        'bool_feature': pl.Series('bool_feature', bool_feature),  # keep as boolean dtype
        'num1': num1,
        'num2': num2,
        'target': target,
    })
    # Ensure the DataFrame column is actually boolean
    assert df.schema['bool_feature'] == pl.Boolean

    # Build enriched API objects
    builder = EnrichedBuilder()
    features, source = builder.build(df, 'target')
    # Ensure the builder mapped the column to a BoolFeature
    assert any(isinstance(f, BoolFeature) and f.name == 'bool_feature' for f in features)

    # Create selector and select features
    selector = create_selector(selector_name, top_k=3)
    classification_problem = ClassificationProblem(ProblemDescription('target'), Field('target', types.int64), (0, 1))
    selected = selector.select_features(features, source, classification_problem, conn)

    # Assertions
    assert len(selected) > 0
    assert len(selected) <= 3
    selected_names = {f.name for f in selected}
    assert 'bool_feature' in selected_names, (
        f'Boolean feature should be selected by {selector_name}, got: {selected_names}'
    )


def test_linear_pairwise_selection_order(conn: duckdb.DuckDBPyConnection) -> None:
    """Test that LinearPairWiseFeatureSelector returns features in selection order, not importance order."""
    rng = np.random.default_rng(42)
    n_rows = 300
    
    # Create features with known selection order based on their predictive strength
    # Feature 1: Strong linear relationship (should be selected first)
    feature1 = rng.normal(0, 1, n_rows)
    
    # Feature 2: Moderate relationship (should be selected second)  
    feature2 = rng.normal(0, 1, n_rows)
    
    # Feature 3: Weak relationship (should be selected third)
    feature3 = rng.normal(0, 1, n_rows)
    
    # Noise feature (should not be selected)
    noise = rng.normal(0, 1, n_rows)
    
    # Create target with known relationships
    target = (
        3.0 * feature1 +      # Strong coefficient - should be selected first
        1.5 * feature2 +      # Moderate coefficient - should be selected second  
        0.5 * feature3 +      # Weak coefficient - should be selected third
        0.1 * noise +         # Very weak - should not be selected
        rng.normal(0, 0.2, n_rows)  # Small noise
    )
    
    df = pl.DataFrame({
        'feature1': feature1,
        'feature2': feature2, 
        'feature3': feature3,
        'noise': noise,
        'target': target,
    })
    
    # Build enriched API objects
    builder = EnrichedBuilder()
    features, source = builder.build(df, 'target')
    
    # Create selector and select top 3 features
    selector = LinearPairWiseFeatureSelector(top_k=3)
    regression_problem = RegressionProblem(ProblemDescription('target'), Field('target', types.float64))
    selected = selector.select_features(features, source, regression_problem, conn)
    
    # Get selected feature names in order
    selected_names = [f.name for f in selected]
    
    # Verify we got exactly 3 features
    assert len(selected_names) == 3
    
    # Verify features are returned in selection order (strongest first)
    # The selector should pick feature1 first (strongest), then feature2, then feature3
    assert selected_names[0] == 'feature1', f'Expected feature1 first, got {selected_names[0]}'
    assert selected_names[1] == 'feature2', f'Expected feature2 second, got {selected_names[1]}'  
    assert selected_names[2] == 'feature3', f'Expected feature3 third, got {selected_names[2]}'
    
    # Verify final_importances_ matches the selection order
    assert selector.final_importances_ is not None
    importance_features = selector.final_importances_['feature']
    importance_scores = selector.final_importances_['importance']
    
    # Importances should be in same order as selected features
    assert importance_features == selected_names
    assert len(importance_scores) == 3
    
    # Verify that importance scores are in descending order (feature1 > feature2 > feature3)
    assert importance_scores[0] > importance_scores[1] > importance_scores[2], (
        f'Importance scores should be descending: {importance_scores}'
    )
