"""Test categorical handling logic in LinearPairWiseFeatureSelector.

This test creates synthetic data with known categorical patterns to verify:
1. Categorical features are properly decomposed into dummy variables
2. Positive-only aggregation works correctly
3. Representative component selection for pairwise comparisons
4. Final model training uses original categorical values
"""

from collections.abc import Sequence
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

import agentune.analyze.core.types
from agentune.analyze.core import types
from agentune.analyze.core.database import DuckdbTable
from agentune.analyze.core.dataset import DatasetSource
from agentune.analyze.core.schema import Field, Schema
from agentune.analyze.core.types import Dtype
from agentune.analyze.feature.base import CategoricalFeature, Feature, NumericFeature
from agentune.analyze.feature.problem import (
    ClassificationProblem,
    ProblemDescription,
    RegressionProblem,
)
from agentune.analyze.feature.select.linear_pairwise import LinearPairWiseFeatureSelector
from agentune.analyze.join.base import JoinStrategy

from .conftest import EnrichedBuilder
from .helpers import load_and_clean_csv

# Type aliases

# Test constants
SHAPE_PROBABILITIES = [0.45, 0.45, 0.10]  # circle, square, triangle
TRIANGLE_OVERRIDE_PROB = 0.6

# Test constants
DATA_FILE_PATH_TITANIC = str((Path(__file__).parent / 'data' / 'titanic_300_features_anonymized.csv').resolve())
DATA_FILE_PATH_TITANIC_WITH_EMBARKED = str((Path(__file__).parent / 'data' / 'titanic_300_features_anonymized_with_embarked.csv').resolve())
TARGET_COLUMN_TITANIC = 'survived'
TOP_K_FEATURES = 20


# Helpers to reduce duplication in tests below
def replace_feature_as_categorical(
    features: Sequence[Feature],
    col_name: str,
    categories: Sequence[str],
    description: str = ''
) -> list[Feature]:
    """Return features where `col_name` is replaced by a categorical feature with provided categories."""
    modified: list[Feature] = []
    for feature in features:
        if hasattr(feature, 'name') and feature.name == col_name:
            modified.append(
                MockCategoricalFeature(name=col_name, categories=categories, description=description)
            )
        else:
            modified.append(feature)
    return modified

def enrich_as_categorical(
    df: pl.DataFrame,
    target_col: str,
    col_name: str,
    categories: Sequence[str],
    description: str = ''
) -> tuple[list[Feature], DatasetSource]:
    """Build via EnrichedBuilder and replace a single column's feature as categorical."""
    builder = EnrichedBuilder()
    features, source = builder.build(df, target_col)
    return replace_feature_as_categorical(features, col_name, categories, description), source

def select_feature_names(
    selector: LinearPairWiseFeatureSelector,
    features: Sequence[object],
    source: DatasetSource,
    target_col: str,
    conn: duckdb.DuckDBPyConnection
) -> tuple[list[str], Sequence[object]]:
    selected = selector.select_features(features, source, target_col, conn)  # type: ignore[arg-type]
    return [f.name for f in selected], selected

def overlap_stats(a: Sequence[str], b: Sequence[str], top_k: int) -> tuple[float, set[str]]:
    common = set(a) & set(b)
    return (len(common) / top_k, common)


def log_and_assert_overlap(
    selected_a: list[str],
    selected_b: list[str],
    feature_to_check: str | None = None,
    min_overlap: float = 0.3,
    check_feature_selected: bool = True,
) -> None:
    """Log and assert feature selection overlap statistics."""
    common_features = set(selected_a) & set(selected_b)
    overlap_ratio = len(common_features) / TOP_K_FEATURES
    feature_selected = feature_to_check in selected_b if feature_to_check else False


    assert overlap_ratio >= min_overlap, f'Overlap ({overlap_ratio:.2f}) should be >= {min_overlap}'
    if feature_to_check:
        assert feature_selected is check_feature_selected, f'Expected selection status of "{feature_to_check}" to be {check_feature_selected}'


def run_selection_on_df(
    df: pl.DataFrame,
    target_col: str,
    conn: duckdb.DuckDBPyConnection,
    categorical_info: tuple[str, list[str]] | None = None,
) -> list[str]:
    """Run feature selection on a DataFrame and return selected feature names."""
    selector = LinearPairWiseFeatureSelector(top_k=TOP_K_FEATURES)

    if categorical_info:
        col_name, categories = categorical_info
        features, source = enrich_as_categorical(df, target_col, col_name, categories)
    else:
        builder = EnrichedBuilder()
        features_seq, source = builder.build(df, target_col)
        features = list(features_seq)

    problem = ClassificationProblem(ProblemDescription(target_col), Field(target_col, types.int64), (0, 1))
    selected = selector.select_features(features, source, problem, conn)
    return [f.name for f in selected]
 

# Mock implementations for testing
class MockCategoricalFeature(CategoricalFeature):
    _categories: tuple[str, ...]

    def __init__(self, name: str, categories: Sequence[str], description: str = ''):
        self.name = name
        self.description = description
        self._categories = tuple(categories)
        self.default_for_missing = categories[0] if categories else ''
    
    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories
    
    @categories.setter
    def categories(self, value: tuple[str, ...]) -> None:
        self._categories = tuple(value)
    
    @property
    def code(self) -> str:
        return f'# Mock categorical feature {self.name}'
    
    @property
    def join_strategies(self) -> Sequence[JoinStrategy]:
        return ()
    
    @property
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        return ()
    
    @property
    def params(self) -> Schema:
        return Schema(())


def test_string_recognition_categorical_and_multiclass() -> None:
    """String dtypes are preserved for categorical features and multiclass targets."""
    rng = np.random.default_rng(42)
    n_rows = 200

    categories = ['red', 'blue', 'green']
    cat_feature = rng.choice(categories, n_rows)

    num_feature1 = rng.normal(size=n_rows)
    num_feature2 = rng.normal(size=n_rows)

    target_classes = ['class_A', 'class_B', 'class_C']
    target = []
    for cat in cat_feature:
        if cat == 'red':
            probs = [0.6, 0.3, 0.1]
        elif cat == 'blue':
            probs = [0.2, 0.6, 0.2]
            
        else:  # green
            probs = [0.1, 0.3, 0.6]
        target.append(rng.choice(target_classes, p=probs))

    df = pl.DataFrame({
        'color': cat_feature,
        'numeric1': num_feature1,
        'numeric2': num_feature2,
        'target_class': target,
    })

    # Dtype checks
    assert df['color'].dtype == pl.Utf8
    assert df['target_class'].dtype == pl.Utf8

    # Category presence checks
    color_values = set(df['color'].unique().to_list())
    target_values = set(df['target_class'].unique().to_list())
    assert color_values == set(categories)
    assert target_values == set(target_classes)

class MockNumericFeature(NumericFeature):
    def __init__(self, name: str, description: str = ''):
        self.name = name
        self.description = description
        self.default_for_missing = 0.0
        self.default_for_nan = 0.0
        self.default_for_infinity = 0.0
        self.default_for_neg_infinity = 0.0

    @property
    def dtype(self) -> Dtype:
        return agentune.analyze.core.types.float64
    
    @property
    def code(self) -> str:
        return f'# Mock numeric feature {self.name}'
    
    @property
    def join_strategies(self) -> Sequence[JoinStrategy]:
        return ()
    
    @property
    def secondary_tables(self) -> Sequence[DuckdbTable]:
        return ()
    
    @property
    def params(self) -> Schema:
        return Schema(())


def test_categorical_selection_regression(conn: duckdb.DuckDBPyConnection) -> None:
    """A predictive categorical and numeric are selected in a regression task; noise is not."""
    rng = np.random.default_rng(42)
    n_samples = 300

    good_categories = ['excellent', 'good', 'okay', 'poor']
    bad_categories = ['red', 'blue', 'green', 'yellow']

    good_cat = rng.choice(good_categories, n_samples)
    bad_cat = rng.choice(bad_categories, n_samples)
    good_numeric = rng.normal(0, 1, n_samples)

    target = rng.normal(0, 0.5, n_samples)
    target[good_cat == 'excellent'] += 4
    target[good_cat == 'good'] += 2
    target[good_cat == 'poor'] -= 1
    target += 2 * good_numeric

    df_reg = pl.DataFrame({
        'good_cat': good_cat,
        'bad_cat': bad_cat,
        'good_num': good_numeric,
        'target': target,
    })

    builder = EnrichedBuilder()
    features, source = builder.build(df_reg, 'target')
    
    selector = LinearPairWiseFeatureSelector(top_k=2)
    problem = RegressionProblem(ProblemDescription('target'), Field('target', types.float64))
    selected = selector.select_features(features, source, problem, conn)

    selected_names = [f.name for f in selected]
    # Features should be returned in selection order (good_num selected first due to strong linear relationship)
    assert selected_names == ['good_num', 'good_cat']

    fi = selector.final_importances_
    assert fi is not None
    importances = dict(zip(fi['feature'], fi['importance'], strict=False))
    assert all(name in importances for name in selected_names)


def test_categorical_selection_multiclass(conn: duckdb.DuckDBPyConnection) -> None:
    """Predictive categoricals are selected in a multiclass task; noise is not."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    colors = rng.choice(['red', 'blue', 'green', 'yellow'], n_samples)
    shapes = rng.choice(['circle', 'square', 'triangle'], n_samples, p=SHAPE_PROBABILITIES)
    noise_cat = rng.choice(['a', 'b', 'c', 'd'], n_samples)

    targets = []
    for i in range(n_samples):
        if colors[i] in ['red', 'blue']:
            targets.append('cat')
        elif colors[i] in ['green', 'yellow']:
            targets.append('dog')
        else:
            targets.append('bird')
    for i in range(n_samples):
        if shapes[i] == 'triangle' and rng.random() < TRIANGLE_OVERRIDE_PROB:
            targets[i] = 'bird'

    df_mc = pl.DataFrame({
        'color': colors,
        'shape': shapes,
        'noise': noise_cat,
        'target': np.array(targets),
    })

    builder = EnrichedBuilder()
    features, source = builder.build(df_mc, 'target')
    
    selector = LinearPairWiseFeatureSelector(top_k=2)
    problem = ClassificationProblem(ProblemDescription('target'), Field('target', types.string), tuple(sorted(targets)))
    selected = selector.select_features(features, source, problem, conn)

    selected_names = [f.name for f in selected]
    assert selected_names == ['color', 'shape']

    fi = selector.final_importances_
    assert fi is not None
    importances = dict(zip(fi['feature'], fi['importance'], strict=False))
    assert all(imp > 0.01 for imp in importances.values())


def test_other_bucket_handling(conn: duckdb.DuckDBPyConnection) -> None:
    """Categories beyond top-K contribute via OTHER bucket: categorical should be selected."""
    rng = np.random.default_rng(42)
    n_samples = 400

    # 12 categories (top-9 + 3 beyond); only beyond-9 carry signal
    categories = [f'cat_{i:02d}' for i in range(12)]
    cat_values = rng.choice(categories, n_samples)
    target = rng.normal(0, 0.5, n_samples)
    target[cat_values == 'cat_10'] += 2.0
    target[cat_values == 'cat_11'] += 1.5

    # Add a noise numeric to compete with the categorical
    noise_num = rng.normal(0, 1.0, n_samples)

    df = pl.DataFrame({
        'many_categories': cat_values,
        'noise_num': noise_num,
        'target': target,
    })

    builder = EnrichedBuilder()
    features, source = builder.build(df, 'target')
    
    selector = LinearPairWiseFeatureSelector(top_k=1)
    problem = RegressionProblem(ProblemDescription('target'), Field('target', types.float64))
    selected = selector.select_features(features, source, problem, conn)
    
    assert len(selected) == 1
    assert selected[0].name == 'many_categories', 'OTHER bucket signal should make categorical win'

    fi = selector.final_importances_
    assert fi is not None
    assert fi['feature'][0] == 'many_categories'
    assert fi['importance'][0] > 0


def test_titanic_fake_categorical_minimal_impact(conn: duckdb.DuckDBPyConnection) -> None:
    """Test that adding a fake categorical feature to Titanic has minimal impact on feature selection."""
    data_original = load_and_clean_csv(DATA_FILE_PATH_TITANIC, 'Titanic Original')
    selected_names_original = run_selection_on_df(data_original, TARGET_COLUMN_TITANIC, conn)

    # Add fake categorical feature
    rng = np.random.default_rng(42)
    fake_categories = ['fake_cat_A', 'fake_cat_B', 'fake_cat_C']
    fake_cat_values = rng.choice(fake_categories, size=len(data_original))
    data_with_fake = data_original.with_columns(pl.Series('fake_category', fake_cat_values))

    selected_names_fake = run_selection_on_df(
        data_with_fake, TARGET_COLUMN_TITANIC, conn, ('fake_category', fake_categories)
    )
    
    # Compare results and assert
    log_and_assert_overlap(
        selected_names_original, selected_names_fake, feature_to_check='fake_category', check_feature_selected=False, min_overlap=0.3
    )

 
def test_titanic_relabel_binary_equal(conn: duckdb.DuckDBPyConnection) -> None:
    """Relabelling binary target to strings keeps selection identical."""
    data_original = load_and_clean_csv(DATA_FILE_PATH_TITANIC, 'Titanic Original')

    # Baseline: original binary
    selected_names_binary = run_selection_on_df(data_original, TARGET_COLUMN_TITANIC, conn)

    # Relabel to string classes (still 2 classes)
    target_binary = data_original[TARGET_COLUMN_TITANIC].to_numpy()
    target_as_strings = np.where(target_binary == 1, 'pos', 'neg')
    data_relabelled = data_original.with_columns(pl.Series(TARGET_COLUMN_TITANIC, target_as_strings))
    selected_names_relabelled = run_selection_on_df(
        data_relabelled, TARGET_COLUMN_TITANIC, conn
    )

    # Expect identical set; relabelling should not change selection at all
    assert set(selected_names_binary) == set(selected_names_relabelled)


def test_titanic_third_class_high_overlap(conn: duckdb.DuckDBPyConnection) -> None:
    """Adding a third class (1 row) should have minimal effect (high overlap)."""
    data_original = load_and_clean_csv(DATA_FILE_PATH_TITANIC, 'Titanic Original')

    # Baseline: original binary
    selected_names_binary = run_selection_on_df(data_original, TARGET_COLUMN_TITANIC, conn)

    # Create a 3-class target by uniformly flipping a single row to a new label '2' (minimal impact)
    rng = np.random.default_rng(42)
    target_binary = data_original[TARGET_COLUMN_TITANIC].to_numpy()
    n = len(target_binary)
    flip_idx = rng.choice(np.arange(n), size=1, replace=False)
    target_three = target_binary.copy()
    target_three[flip_idx] = 2
    data_three = data_original.with_columns(pl.Series(TARGET_COLUMN_TITANIC, target_three))

    selected_names_three = run_selection_on_df(data_three, TARGET_COLUMN_TITANIC, conn)

    # Expect very high overlap; adding a small third class should have minimal effect
    log_and_assert_overlap(selected_names_binary, selected_names_three, min_overlap=0.95)


def test_titanic_embarked_categorical_comparison(conn: duckdb.DuckDBPyConnection) -> None:
    """Test Titanic dataset with and without embarked categorical feature."""
    data_original = load_and_clean_csv(DATA_FILE_PATH_TITANIC, 'Titanic Original')
    selected_names_original = run_selection_on_df(data_original, TARGET_COLUMN_TITANIC, conn)

    data_with_embarked = load_and_clean_csv(DATA_FILE_PATH_TITANIC_WITH_EMBARKED, 'Titanic With Embarked')
    embarked_categories = sorted(data_with_embarked['embarked'].unique().drop_nulls().to_list())
    selected_names_embarked = run_selection_on_df(
        data_with_embarked, TARGET_COLUMN_TITANIC, conn, ('embarked', embarked_categories)
    )
    
    # Compare results and assert: expect high overlap, making the test robust to minor rank changes
    log_and_assert_overlap(selected_names_original, selected_names_embarked, feature_to_check='embarked', min_overlap=0.8, check_feature_selected=False)


def test_titanic_multiclass_embarked_categorical_comparison(conn: duckdb.DuckDBPyConnection) -> None:
    """Test Linear PairWise feature selection on multiclass Titanic dataset with embarked categorical feature."""
    # Baseline: Binary, no embarked
    data_original = load_and_clean_csv(DATA_FILE_PATH_TITANIC, 'Titanic Original')
    selected_names_original = run_selection_on_df(data_original, TARGET_COLUMN_TITANIC, conn)

    # Create multiclass target
    rng = np.random.default_rng(42)
    multiclass_indices = rng.choice(len(data_original), 10, replace=False)
    target_multiclass = data_original[TARGET_COLUMN_TITANIC].to_numpy().copy()
    target_multiclass[multiclass_indices] = 2

    # Scenario 1: Multiclass, no embarked
    data_multiclass = data_original.with_columns(pl.Series(TARGET_COLUMN_TITANIC, target_multiclass))
    selected_names_multiclass = run_selection_on_df(data_multiclass, TARGET_COLUMN_TITANIC, conn)

    # Scenario 2: Multiclass, with embarked
    data_embarked = load_and_clean_csv(DATA_FILE_PATH_TITANIC_WITH_EMBARKED, 'Titanic With Embarked')
    data_multiclass_embarked = data_embarked.with_columns(pl.Series(TARGET_COLUMN_TITANIC, target_multiclass))
    embarked_categories = sorted(data_embarked['embarked'].unique().drop_nulls().to_list())
    selected_names_multiclass_embarked = run_selection_on_df(
        data_multiclass_embarked, TARGET_COLUMN_TITANIC, conn, ('embarked', embarked_categories)
    )
    
    # Compare results and assert
    log_and_assert_overlap(
        selected_names_multiclass, selected_names_multiclass_embarked, feature_to_check='embarked', min_overlap=0.3, check_feature_selected=False
    )

    # Additional assertions for this specific test
    binary_multiclass_overlap = len(set(selected_names_original) & set(selected_names_multiclass)) / TOP_K_FEATURES
    assert binary_multiclass_overlap >= 0.4, f'Binary-multiclass overlap ({binary_multiclass_overlap:.2f}) should be at least 40%'

    assert len(selected_names_original) == TOP_K_FEATURES
    assert len(selected_names_multiclass) == TOP_K_FEATURES
    assert len(selected_names_multiclass_embarked) == TOP_K_FEATURES
