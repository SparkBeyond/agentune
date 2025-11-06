import asyncio
from collections.abc import Sequence

import attrs
import polars as pl
from attrs import frozen

from agentune.analyze.api.base import RunContext
from agentune.analyze.api.data import (
    BoundDatasetSink,
    BoundDatasetSource,
    BoundSplitTable,
    BoundTable,
)
from agentune.analyze.core.database import DuckdbName, DuckdbTable
from agentune.analyze.core.dataset import Dataset, DatasetSource
from agentune.analyze.core.schema import Schema
from agentune.analyze.feature.base import Feature
from agentune.analyze.feature.problem import ProblemDescription
from agentune.analyze.feature.recommend import ActionRecommender, RecommendationsReport
from agentune.analyze.join.base import JoinStrategy, TablesWithJoinStrategies
from agentune.analyze.run.analysis.base import (
    AnalyzeComponents,
    AnalyzeInputData,
    AnalyzeParams,
    AnalyzeResults,
    AnalyzeRunner,
)
from agentune.analyze.run.ingest.sampling import SplitDuckdbTable


@frozen
class BoundOps:
    """Methods for running high-level operations, bound to a context instance."""
    run_context: RunContext

    async def analyze(self, problem_description: ProblemDescription,
                      main_input: AnalyzeInputData | SplitDuckdbTable | BoundSplitTable,
                      test_input: str | DuckdbName | DuckdbTable | BoundTable | DatasetSource | BoundDatasetSource | Dataset | pl.DataFrame | None = None,
                      secondary_tables: Sequence[str | DuckdbName | DuckdbTable | BoundTable] = (),
                      join_strategies: Sequence[JoinStrategy] = (),
                      params: AnalyzeParams | None = None,
                      runner: AnalyzeRunner | None = None,
                      components: AnalyzeComponents | None = None) -> AnalyzeResults:
        """Generate, evaluate and select features.

        The analysis process proceeds roughly as follows:
        - Validate the input and determine the problem characteristics. E.g., decide if it is a regression or classification
          problem (if not specified by the user) and collect the target class values.
          An instance of class `Problem` is created; it incorporates the `problem_description`.
        - Generate candidate features using the feature generator(s) provided.
          The generators use the feature search data split of the main input (or a subsample of it)
          and the specified secondary tables and join strategies.
        - Enrich the entire feature search data split with the candidate features, and calculate statistics
          (class `FullFeatureStats`) for each candidate feature.
        - Based on these statistics and other feature metadata, select the final features using the feature_selector component.
        - Enrich the feature evaluation and test data splits with the selected features.

        Args:
            problem_description: defines the target column and allows providing optional metadata about the problem
                                 and data.
            main_input: the primary input data, split into four parts (train, test, feature search and feature evaluation).
                        Passing a AnalyzeInputData instance allows full control over the splits; in this case
                        the required invariants are not checked, e.g. the train and test can overlap.
                        This also allows passing in DatasetSources for some of the splits that read an external data source
                        and not a duckdb table; this is not recommended, and good performance cannot be guaranteed.
            test_input: If not None, this overrides the test split given in the main_input.
                        Note that you can split a table with a train fraction of 1.0, creating an empty test split,
                        if you intend to pass a separate test data source.

                        A string or DuckdbName names a table to read from the database; other types provide input directly.
            secondary_tables: duckdb tables other than the main input that features can access.
                              Unlike the main input, secondary tables are always considered in their entirety and
                              are not split into train and test.
            join_strategies: optional predefined strategies for joining the secondary tables to the main input.
                             Features are not limited by the join strategies in the ways they use the secondary tables,
                             but some feature generators require join strategies to be specified.
            params:          all other parameters affecting the analyzer.
            runner:          a AnalyzeRunner instance, allowing to use a custom or wrapped implementation.
                             If None, the default is used, given by ctx.defaults.analyze_runner().
            components:      instances of all other components (feature generators, the feature selector, etc)
                             used by the analyzer. If None, the default is used, given by
                             ctx.defaults.analyze_components().
        """
        tables_with_join_strategies = TablesWithJoinStrategies.unflatten(
            [self._ref_to_table(ref) for ref in secondary_tables], join_strategies)

        with self.run_context._ddb_manager.cursor() as conn:
            match main_input:
                case SplitDuckdbTable() as split:
                    input_data = AnalyzeInputData.from_split_table(split, tables_with_join_strategies, conn)
                case BoundSplitTable() as split:
                    input_data = AnalyzeInputData.from_split_table(split.splits, tables_with_join_strategies,
                                                                         conn)
                case AnalyzeInputData() as inputs:
                    if len(join_strategies) > 0 and inputs.join_strategies != tables_with_join_strategies:
                        raise ValueError('Values of params join_strategies and main_input.join_strategies must match. '
                                         'You can leave the parameter join_strategies empty if you want to use the value '
                                         'given in main_input.')
                    input_data = inputs

            if test_input is not None:
                input_data = attrs.evolve(input_data, test=self._input_to_data_source(test_input))

            params = params or self.run_context.defaults.analyze_params()
            components = components or self.run_context.defaults.analyze_components()
            runner = runner or self.run_context.defaults.analyze_runner()
            return await runner.run(self.run_context._ddb_manager, input_data, params, components, problem_description)

    def _ref_to_table(self, ref: str | DuckdbName | DuckdbTable | BoundTable) -> DuckdbTable:
        match ref:
            case str() as name:
                return self.run_context.db.table(name).table
            case DuckdbName() as name:
                return self.run_context.db.table(name).table
            case DuckdbTable() as table:
                return table
            case BoundTable() as ref:
                return ref.table
            case _:
                raise TypeError(f'Invalid table specification: {ref}')
        
    def _input_to_data_source(self, source: str | DuckdbName | DuckdbTable | BoundTable | DatasetSource | BoundDatasetSource | Dataset | pl.DataFrame) -> DatasetSource:
        match source:
            case str() | DuckdbName() | DuckdbTable() | BoundTable() as ref:
                return DatasetSource.from_table(self._ref_to_table(ref))
            case DatasetSource() as dataset_source:
                return dataset_source
            case BoundDatasetSource() as dataset_source:
                return dataset_source.dataset_source
            case Dataset() as dataset:
                return DatasetSource.from_dataset(dataset)
            case pl.DataFrame() as df:
                return DatasetSource.from_dataset(Dataset.from_polars(df))

    def _canonicalize_keep_input_columns(self, schema: Schema,
                                         keep_input_columns: Sequence[str] | bool) -> Sequence[str]:
        match keep_input_columns:
            case True:
                return schema.names
            case False:
                return ()
            case names:
                return names

    async def enrich(self, input: Dataset | pl.DataFrame, features: Sequence[Feature],
                     keep_input_columns: Sequence[str] | bool = False,
                     deduplicate_names: bool = True) -> Dataset:
        """Enrich a dataset using previously found features. One column per feature will be added.

        The secondary tables used by these features must be present under the same names as during the analysis when the
        features were created.

        Args:
             input: a dataset with a schema compatible with that of the main input during the analysis when these
                    features were constructed.
                    Columns present in the original main input that are not used by any of the given features can be omitted.
                    The columns' order does not have to match the original order.
             features: the features to compute. One column will be added to the output for each feature.
                       The name of each output column will be the `feature.name`.
             keep_input_columns: whether to include the input columns in the output. If False, only the generated feature
                                 columns appear in the output.
             deduplicate_names: If some features have the same name, or if a feature has the same name as one of the input
                                columns (and keep_input_columns is True), then: if this parameter is True, the generated
                                features columns will be renamed to avoid collisions; if it is False, an error will be raised.
        """
        if isinstance(input, pl.DataFrame):
            input = Dataset.from_polars(input)
        evaluators = self.run_context.defaults.feature_computers()
        keep_input_columns = self._canonicalize_keep_input_columns(input.schema, keep_input_columns)
        with self.run_context._ddb_manager.cursor() as conn:
            return await self.run_context.defaults.enrich_runner().run(features, input, evaluators, conn,
                                                                       keep_input_columns, deduplicate_names)

    async def enrich_stream(self, features: Sequence[Feature],
                            input: str | DuckdbName | DuckdbTable | BoundDatasetSource | DatasetSource | Dataset | pl.DataFrame,
                            output: BoundDatasetSink,
                            keep_input_columns: Sequence[str] | bool = False,
                            deduplicate_names: bool = True) -> None:
        """Enrich a data stream using previously found features, writing it to the output.

        This behaves the same as `enrich` (see there), but can read any DatasetSource (including external data)
        rather than a Dataset present in memory, and writes to an arbitrary DatasetSink. The data is processed
        in a streaming fashion.

        Args:
            features: the features to compute. One column will be added to the output for each feature.
                       The name of each output column will be the `feature.name`.
            input: the data to enrich.
                   A string or DuckdbName refers to a table in the database.
            output: the location to write the enriched data.
            keep_input_columns: whether to include the input columns in the output. If False, only the generated feature
                                columns appear in the output.
            deduplicate_names: If some features have the same name, or if a feature has the same name as one of the input
                               columns (and keep_input_columns is True), then: if this parameter is True, the generated
                               features columns will be renamed to avoid collisions; if it is False, an error will be raised.
        """
        dataset_source = self._input_to_data_source(input)
        evaluators = self.run_context.defaults.feature_computers()
        keep_input_columns = self._canonicalize_keep_input_columns(dataset_source.schema, keep_input_columns)
        with self.run_context._ddb_manager.cursor() as conn:
            await self.run_context.defaults.enrich_runner().run_stream(features, dataset_source,
                                                                       output.dataset_sink, evaluators,
                                                                       conn, keep_input_columns, deduplicate_names)

    async def recommend_conversation_actions(self,
                                             analyze_input: AnalyzeInputData | SplitDuckdbTable | BoundSplitTable,
                                             analyze_results: AnalyzeResults) -> RecommendationsReport | None:
        return await self.recommend_actions(analyze_input, analyze_results,
                                            self.run_context.defaults.conversation_action_recommender())

    async def recommend_actions(self,
                                analyze_input: AnalyzeInputData | SplitDuckdbTable | BoundSplitTable,
                                analyze_results: AnalyzeResults,
                                recommender: ActionRecommender) -> RecommendationsReport | None:
        """Recommend actions using the inputs and results of an analysis run.

        Args:
            analyze_input: the original inputs to the analysis.
                           Only the feature search split is used; this argument type is meant for the convenience
                           of calling this method immediately after an analysis run.
            analyze_results: the results of the analysis run.
            recommender:     the action recommender to use. Different recommenders support different features and
                             produce different report types.

        Returns:
            None if the given recommender is unable to produce a report for the given problem and features.
        """
        match analyze_input:
            case AnalyzeInputData() as input_data:
                dataset_source = input_data.feature_eval
            case SplitDuckdbTable() as split:
                dataset_source = split.feature_eval()
            case BoundSplitTable() as split:
                dataset_source = split.splits.feature_eval()
        with self.run_context._ddb_manager.cursor() as conn:
            dataset = await asyncio.to_thread(dataset_source.copy_to_thread().to_dataset, conn)
        with self.run_context._ddb_manager.cursor() as conn:
            return await recommender.arecommend(analyze_results.problem, analyze_results.features_with_train_stats, dataset, conn)
