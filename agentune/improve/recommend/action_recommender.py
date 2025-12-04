"""Action recommender using LLM to analyze conversation data.

This recommender formats conversation data and uses an LLM to generate
actionable recommendations based on feature importance.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import ClassVar, override

import attrs
import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.feature.gen.insightful_text_generator.features import InsightfulTextFeature
from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import DataFormatter
from agentune.analyze.feature.gen.insightful_text_generator.formatting.conversation import (
    ShortDateConversationFormatter,
)
from agentune.analyze.feature.gen.insightful_text_generator.sampling.base import (
    DataSampler,
)
from agentune.analyze.feature.gen.insightful_text_generator.sampling.samplers import (
    BalancedClassSampler,
    BalancedNumericSampler,
)
from agentune.analyze.feature.gen.insightful_text_generator.util import (
    achat_raw,
    estimate_tokens,
    get_max_input_context_window,
)
from agentune.analyze.feature.problem import (
    ClassificationProblem,
    Problem,
    RegressionProblem,
)
from agentune.analyze.feature.stats.base import FeatureWithFullStats
from agentune.analyze.join.conversation import Conversation, ConversationJoinStrategy
from agentune.core.dataset import Dataset
from agentune.core.llm import LLMContext, LLMSpec
from agentune.core.sercontext import LLMWithSpec
from agentune.improve.recommend import prompts
from agentune.improve.recommend.base import ActionRecommender
from agentune.improve.recommend.prompts import ConversationVerification

logger = logging.getLogger(__name__)

MIN_SAMPLES = 5  # Minimum samples for formatting conversations

@frozen
class FeatureWithScore:
    """Reference to a feature with its importance score."""
    name: str
    r_squared: float


@frozen
class ConversationWithExplanation:
    """A conversation reference with explanation of relevance."""
    conversation_id: int
    explanation: str


@frozen
class ConversationWithMetadata:
    """A conversation with its metadata for the report.
    
    This is stored in a dict keyed by display_number (the number shown in the LLM prompt).
    
    Fields:
    - actual_id: The actual database conversation ID (UUID, string, etc.)
    - conversation: The conversation content itself
    - outcome: The outcome/target value for this conversation (optional)
    """
    actual_id: int | str
    conversation: Conversation
    outcome: str


@frozen
class Recommendation:
    """An actionable recommendation with enriched feature data."""
    title: str
    description: str
    rationale: str
    evidence: str
    supporting_features: tuple[FeatureWithScore, ...]
    supporting_conversations: tuple[ConversationWithExplanation, ...]


@frozen
class RecommendationsReport:
    """Structured recommendations report with enriched R² data."""
    analysis_summary: str
    recommendations: tuple[Recommendation, ...]
    conversations: dict[int, ConversationWithMetadata]  # Keyed by display_number (1, 2, 3...)
    raw_report: str
    total_conversations_analyzed: int  # Total number of conversations sent to LLM for analysis
    all_features: tuple[FeatureWithScore, ...]  # All features analyzed, sorted by R²


@attrs.frozen
class ConversationActionRecommender(ActionRecommender):
    """Recommender that uses conversation data and LLM to generate actionable recommendations.
    
    This recommender filters conversation features dynamically and generates
    recommendations based on their importance. Currently supports a single
    conversation strategy per recommendation.
    
    The recommender:
    1. Discovers features that use conversations
    2. Groups by ConversationJoinStrategy
    3. Validates single strategy (raises error for multiple)
    4. Samples data using problem-appropriate sampler:
       - Classification: BalancedClassSampler (equal samples per class)
       - Regression: BalancedNumericSampler (equal samples per quantile bin)
    5. Formats conversations for LLM input, adapting number of examples to token limit
    6. Generates LLM-based recommendations
    """

    default_model: ClassVar[LLMSpec] = LLMSpec('openai', 'o3')
    default_structuring_model: ClassVar[LLMSpec] = LLMSpec('openai', 'gpt-4o')

    @staticmethod
    def default(llm_context: LLMContext) -> 'ConversationActionRecommender':
        return ConversationActionRecommender(
            model=LLMWithSpec(ConversationActionRecommender.default_model, llm_context.from_spec(ConversationActionRecommender.default_model)),
            structuring_model=LLMWithSpec(ConversationActionRecommender.default_structuring_model, llm_context.from_spec(ConversationActionRecommender.default_structuring_model)),
        )

    model: LLMWithSpec
    max_samples: int = 40
    top_k_features: int = 60
    
    # Optional faster model for structuring (e.g., gpt-4o instead of o3)
    structuring_model: LLMWithSpec | None = None
    
    # Descriptions for the agent and instances being analyzed (defaults defined in prompts.py)
    agent_description: str = prompts.DEFAULT_AGENT_DESCRIPTION

    def _get_sampler(self, problem: Problem) -> DataSampler:
        """Get appropriate sampler based on problem type.
        
        Uses balanced sampling strategy:
        - Classification: BalancedClassSampler (equal samples per class)
        - Regression: BalancedNumericSampler (equal samples per quantile bin)
        """
        match problem:
            case ClassificationProblem():
                return BalancedClassSampler(target_field=problem.target_column)
            case RegressionProblem():
                return BalancedNumericSampler(
                    target_field=problem.target_column,
                    num_bins=5,  # Same as analyze default
                )
            case _:
                raise ValueError(f'Unsupported problem type: {type(problem)}')

    @staticmethod
    def _find_and_prepare_conv_features(
            features_with_stats: Sequence[FeatureWithFullStats]
    ) -> tuple[list[FeatureWithFullStats], ConversationJoinStrategy] | None:
        """Find conversation features, validate single strategy, and return sorted features with strategy.

        Args:
            features_with_stats: All features with their statistics

        Returns:
            Tuple of (sorted features by R squared, conversation strategy) if conversation features found.
            None if no conversation features are found.
            
        Raises:
            ValueError: If multiple conversation strategies are found (not yet supported)
        """
        # Collect all conversation features with their strategies
        conv_features: list[tuple[FeatureWithFullStats, ConversationJoinStrategy]] = []
        
        for fws in features_with_stats:
            if isinstance(fws.feature, InsightfulTextFeature):
                strategy = fws.feature.join_strategies[0]
                if isinstance(strategy, ConversationJoinStrategy):
                    conv_features.append((fws, strategy))
                else:
                    raise ValueError(f'Feature {fws.feature.name} has non-conversation join strategy: {strategy}')

        # No conversation features found
        if not conv_features:
            return None

        # Validate single strategy by checking all strategies are the same
        first_strategy = conv_features[0][1]
        unique_strategies = {strategy.name for _, strategy in conv_features}
        if len(unique_strategies) > 1:
            raise ValueError(
                f'Found {len(unique_strategies)} conversation sources: {list(unique_strategies)}. '
                f'Multiple conversation sources not yet supported.'
            )
        
        # Sort features by R squared (importance)
        sorted_features = sorted(
            [fws for fws, _ in conv_features],
            key=lambda fws: fws.stats.relationship.r_squared,
            reverse=True
        )
        
        return sorted_features, first_strategy

    def _get_formatter(self, conversation_strategy: ConversationJoinStrategy, problem: Problem, is_batch: bool) -> DataFormatter:
        params_to_print = (problem.target_column,) if is_batch else ()
        return ShortDateConversationFormatter(
            name=f'conversation_formatter_{conversation_strategy.name}',
            conversation_strategy=conversation_strategy,
            params_to_print=params_to_print,
            include_in_batch_id=is_batch
        )

    def _format_r_squared_dict(
        self, features_with_stats: list[FeatureWithFullStats]
    ) -> str:
        """Format R squared dictionary as a readable string.

        Args:
            features_with_stats: Features sorted by importance
            
        Returns:
            Formatted string showing feature descriptions and R squared values
        """
        lines = []
        for i, fws in enumerate(features_with_stats[:self.top_k_features], 1):
            r_squared = fws.stats.relationship.r_squared
            description = fws.feature.description
            lines.append(f'{i}. {description}: {r_squared:.4f}')
        return '\n'.join(lines)

    async def _verify_conversation_references(
        self,
        report: RecommendationsReport,
        formatted_conversations: pl.Series,
    ) -> RecommendationsReport:
        """Verify conversation references using LLM and filter unsupported ones.

        Args:
            report: RecommendationsReport containing supporting_conversations for each recommendation
            formatted_conversations: Series of formatted conversation strings keyed by display_number (1-based)

        Returns:
            Filtered RecommendationsReport containing recommendations with verified supporting_conversations only
        """

        async def verify_single_conversation(
            rec: Recommendation,
            conv_with_exp: ConversationWithExplanation,
        ) -> bool:
            """Verify a single conversation against a recommendation.

            Args:
                rec: The recommendation being verified
                conv_with_exp: Conversation with explanation to verify

            Returns:
                True if conversation supports the recommendation as claimed, False otherwise
            """
            # Check if conversation ID exists in formatted map
            if conv_with_exp.conversation_id > len(formatted_conversations) or conv_with_exp.conversation_id < 1:
                logger.warning(
                    f'Conversation {conv_with_exp.conversation_id} for recommendation {rec.title} could not be verified')
                return False  # remove conversation if ID not found

            conversation_formatted = formatted_conversations[conv_with_exp.conversation_id - 1]

            # Get outcome from report.conversations
            outcome = report.conversations[conv_with_exp.conversation_id].outcome

            try:
                result = await self.model.llm.astructured_predict(
                    output_cls=ConversationVerification,
                    prompt=prompts.CONVERSATION_VERIFICATION_PROMPT,
                    title=rec.title,
                    description=rec.description,
                    rationale=rec.rationale,
                    evidence=rec.evidence,
                    explanation=conv_with_exp.explanation,
                    conversation_id=conv_with_exp.conversation_id,
                    conversation=conversation_formatted,
                    outcome=outcome,
                )
                return result.supports_recommendation
            except Exception as e: # noqa: BLE001 - catch all LLM/API error, return True as default
                logger.warning(
                    f'Failed to verify conversation {conv_with_exp.conversation_id} '
                    f'for recommendation "{rec.title}": {e}. '
                )
                return True  # keep conversation if verification fails

        # Flatten all (recommendation, conversation) pairs across all recommendations
        items = [
            (rec, conv_with_exp)
            for rec in report.recommendations
            for conv_with_exp in rec.supporting_conversations
        ]

        # Verify ALL conversations in parallel
        results = await asyncio.gather(*[
            verify_single_conversation(rec, conv_with_exp)
            for rec, conv_with_exp in items
        ])

        # Group all verification results by recommendation
        verified_by_rec: dict[Recommendation, dict[ConversationWithExplanation, bool]] = {}
        for (rec, conv_with_exp), is_valid in zip(items, results, strict=True):
            if rec not in verified_by_rec:
                verified_by_rec[rec] = {}
            verified_by_rec[rec][conv_with_exp] = is_valid

        # Clean evidence and rationale in parallel for all recommendations
        async def clean_fields_for_rec(rec: Recommendation) -> tuple[str, str]:
            """Clean evidence and rationale by removing unverified conversation references."""
            # Filter for unverified conversations
            unverified_ids = {
                conv.conversation_id
                for conv, is_valid in verified_by_rec.get(rec, {}).items()
                if not is_valid
            }
            return await prompts.clean_evidence_and_rationale_with_llm(
                evidence=rec.evidence,
                rationale=rec.rationale,
                unverified_conversation_ids=unverified_ids,
                model=self.model,
                structuring_model=self.structuring_model,
            )

        # Clean all evidence and rationale fields in parallel
        logger.info('Starting evidence and rationale verification...')
        cleaned_fields_list = await asyncio.gather(*[
            clean_fields_for_rec(rec)
            for rec in report.recommendations
        ])

        # Rebuild recommendations
        filtered_recommendations = []
        for rec, (cleaned_evidence, cleaned_rationale) in zip(report.recommendations, cleaned_fields_list, strict=True):
            # Filter for verified conversations
            verified_conversations = tuple(
                conv for conv, is_valid in verified_by_rec.get(rec, {}).items() if is_valid
            )

            if cleaned_evidence != rec.evidence:
                logger.debug(f'Evidence for recommendation {rec.title} has been changed due to conversations references verification. '
                             f'Original evidence: {rec.evidence}, current evidence: {cleaned_evidence}')

            if cleaned_rationale != rec.rationale:
                logger.debug(
                    f'Rationale for recommendation {rec.title} has been changed due to conversations references verification. '
                    f'Original rationale: {rec.rationale}, current rationale: {cleaned_rationale}')

            # Only create new Recommendation if conversations changed
            if len(verified_conversations) != len(rec.supporting_conversations):
                filtered_recommendations.append(
                    attrs.evolve(
                        rec,
                        evidence=cleaned_evidence,
                        rationale=cleaned_rationale,
                        supporting_conversations=verified_conversations,
                    )
                )
            else:
                # No change - keep original recommendation
                filtered_recommendations.append(rec)

        return attrs.evolve(report,
                            recommendations=tuple(filtered_recommendations))

    def _convert_pydantic_to_attrs(
        self,
        pydantic_report: prompts.StructuredReport,
        features_with_stats: list[FeatureWithFullStats],
        conversations: tuple[Conversation | None, ...],
        conversation_ids: list[int | str],
        outcomes: list[str],
        raw_report: str,
    ) -> RecommendationsReport:
        """Convert Pydantic report to attrs report, enriching with R squared and conversation data.
        non-existing features appear with R_squared=0, and are to be pruned in the filtering later on in the flow.

        Args:
            pydantic_report: The Pydantic model from LLM structured output
            features_with_stats: Features with their statistics (for R squared lookup)
            conversations: Tuple of Conversation objects (in same order as shown to LLM)
            conversation_ids: List of actual database conversation IDs (in same order as conversations)
            outcomes: List of outcome values (in same order as conversations)
            raw_report: Raw report from LLM structured output

        Returns:
            RecommendationsReport with enriched feature references and ConversationWithMetadata objects
        """
        if len(conversations) != len(conversation_ids) or len(conversations) != len(outcomes):
            raise ValueError('Length of conversations, conversation_ids, and outcomes must be the same')
        # Build a lookup map: feature description -> R²
        r_squared_lookup = {
            fws.feature.description: fws.stats.relationship.r_squared
            for fws in features_with_stats
        }
        
        def find_r_squared_for_feature(feat_name: str) -> float:
            """Find R² for a feature name.
            
            The LLM is instructed to return exact feature descriptions without R² values.
            We try exact match first, then fallback to fuzzy matching for robustness.
            Returns 0.0 if feature is not found (treated as non-predictive).
            """
            # Try exact match first (expected case)
            if feat_name in r_squared_lookup:
                return r_squared_lookup[feat_name]
            
            # Fallback: try to find by prefix match (in case LLM didn't follow instructions)
            for desc, r2 in r_squared_lookup.items():
                if feat_name.startswith(desc) or desc.startswith(feat_name):
                    return r2
            
            return 0.0  # Feature not found - treat as non-predictive
        
        # Convert all recommendations from Pydantic to attrs
        recommendations_list = []
        all_conversation_indices: set[int] = set()

        for rec in pydantic_report.recommendations:
            # Enrich supporting features with R squared values
            supporting_features = tuple(
                FeatureWithScore(
                    name=feat_name,
                    r_squared=find_r_squared_for_feature(feat_name),
                )
                for feat_name in rec.supporting_features
            )

            # Collect conversation indices from this recommendation
            all_conversation_indices.update(
                conv_ref.conversation_id for conv_ref in rec.supporting_conversations
            )

            recommendations_list.append(
                Recommendation(
                    title=rec.title,
                    description=rec.description,
                    rationale=rec.rationale,
                    evidence=rec.evidence,
                    supporting_features=supporting_features,
                    supporting_conversations=tuple(
                        ConversationWithExplanation(
                            conversation_id=conv_ref.conversation_id,
                            explanation=conv_ref.explanation,
                        )
                        for conv_ref in rec.supporting_conversations
                    ),
                )
            )
        
        # Build conversations dict with metadata for only the referenced conversations
        conversations_dict = {
            idx: ConversationWithMetadata(
                actual_id=conversation_ids[idx - 1],  # idx is 1-based, list is 0-based
                conversation=conversations[idx - 1],  # type: ignore[arg-type]  # idx is 1-based, list is 0-based
                outcome=outcomes[idx - 1],  # idx is 1-based, list is 0-based
            )
            for idx in all_conversation_indices
            if 1 <= idx <= len(conversation_ids)  # Validate idx is within valid range
        }

        # Create list of all features with their R² scores
        all_features_list = [
            FeatureWithScore(
                name=fws.feature.description,
                r_squared=fws.stats.relationship.r_squared,
            )
            for fws in features_with_stats
        ]

        return RecommendationsReport(
            analysis_summary=pydantic_report.analysis_summary,
            recommendations=tuple(recommendations_list),
            conversations=conversations_dict,
            raw_report=raw_report,
            total_conversations_analyzed=len(conversations),
            all_features=tuple(all_features_list),
        )

    def _filter_recommendations_with_no_supporting_features(self, report: RecommendationsReport) -> RecommendationsReport:
        """Filter out recommendations with no valid supporting features.

        Removes:
        - Features with R squared = 0.0 (non-existent or non-predictive) from supporting features list in recommendations.
        - Recommendations that have no remaining features after filtering

        Args:
            report: The unfiltered RecommendationsReport

        Returns:
            Filtered RecommendationsReport with only meaningful recommendations
        """
        filtered_recommendations_list = []
        filtered_out: list[str] = []

        for rec in report.recommendations:
            # Filter out features with R squared=0 (non-existent or non-predictive)
            meaningful_features = tuple(
                f for f in rec.supporting_features if f.r_squared > 0.0
            )

            # Only include recommendation if it has at least one meaningful feature
            if meaningful_features:
                filtered_recommendations_list.append(
                    attrs.evolve(rec, supporting_features=meaningful_features)
                )
            else:
                filtered_out.append(rec.title)

        # Log filtered recommendations
        if filtered_out:
            logger.warning(
                f'Filtered out {len(filtered_out)} recommendation(s) due to having no valid supporting features: '
                f'{", ".join(filtered_out)}'
            )

        return attrs.evolve(report, recommendations=tuple(filtered_recommendations_list))

    async def _create_prompt(self,
                             problem: Problem,
                             sorted_features: list[FeatureWithFullStats],
                             sampled_data: Dataset,
                             conversation_strategy: ConversationJoinStrategy,
                             conn: DuckDBPyConnection,
                             ) -> tuple[str, int]:
        """ Create prompt for LLM based on formatted conversations and token limits.

        Args:
            problem: The problem definition
            sorted_features: Features sorted by importance
            sampled_data: Sampled dataset for examples
            conversation_strategy: The conversation join strategy being used
            conn: Database connection
        Returns:
            Tuple of (prompt string, num_example_used)

        Note:
            This method adapts the number of examples included in the prompt
            to fit within the model's context window token limit.
            it used `sampled_data[:num_example_used]` as the conversations included in the prompt.
        """
        # Create the formatter
        formatter = self._get_formatter(conversation_strategy, problem, is_batch=True)
        formatted_samples = await formatter.aformat_batch(sampled_data, conn)

        # Get token limits
        max_context_window = get_max_input_context_window(self.model)
        # Find the maximum number of examples that fit within token limit
        num_examples = len(formatted_samples)
        while num_examples >= MIN_SAMPLES:
            # Try with current number of examples
            current_examples = formatted_samples.to_list()[:num_examples]

            # Build prompt (adapts to regression vs classification)
            formatted_conversations = '\n'.join(current_examples)
            if formatter.description is None:
                raise ValueError(f'Formatter {formatter.name} description not available')
            prompt = prompts.create_conversation_analysis_prompt(
                agent_description=self.agent_description,
                instance_description=formatter.description,
                problem=problem,
                r_squared_dict=self._format_r_squared_dict(sorted_features),
                conversations=formatted_conversations,
            )

            # Estimate tokens
            current_tokens = estimate_tokens(prompt, self.model)

            # If fits, return prompt and conversations
            if current_tokens <= max_context_window:
                logger.info(
                    f'Fitted prompt with {num_examples} out of {len(formatted_samples)} examples, '
                    f'using {current_tokens} tokens which is < token limit ({max_context_window} tokens)'
                )
                return prompt, num_examples

            # Remove one example and try again
            num_examples -= 1
        # We couldn't fit even with minimum samples
        raise ValueError(f'Cannot fit the minimal amount of needed sample conversations ({MIN_SAMPLES}) within the LLM token limit ({max_context_window}). Try providing shorter conversations as input.')

    @override
    async def arecommend(
        self,
        problem: Problem,
        features_with_stats: Sequence[FeatureWithFullStats],
        dataset: Dataset,
        conn: DuckDBPyConnection,
    ) -> RecommendationsReport | None:
        """Generate actionable recommendations using LLM.
        
        Args:
            problem: The problem definition
            features_with_stats: Features with their statistics
            dataset: The dataset to sample from
            conn: Database connection
            
        Returns:
            RecommendationsReport with structured analysis and recommendations,
            or None if no conversation features found.
            
        Raises:
            ValueError: If multiple conversation strategies are found (not yet supported)
        """
        # 1. Find and prepare conversation features
        conv_result = self._find_and_prepare_conv_features(features_with_stats)
        if conv_result is None:
            return None  # No conversation features - explainer not applicable
        
        sorted_features, conversation_strategy = conv_result

        # 2. Sample data
        # Filter dataset to only rows where conversation exists
        existing_ids = conversation_strategy.ids_exist(dataset, conn)
        filtered_dataset = Dataset(schema=dataset.schema, data=dataset.data.filter(existing_ids))

        sampler = self._get_sampler(problem)
        sampled_data = sampler.sample(filtered_dataset, self.max_samples)

        # 3. create prompt and get conversations used (adapting number of examples to token limit)
        prompt, num_examples_used = await self._create_prompt(
            problem, sorted_features, sampled_data, conversation_strategy, conn
        )
        # Adjust sampled_data to only include the used examples
        sampled_data = Dataset(schema=sampled_data.schema, data=sampled_data.data[:num_examples_used])

        # 4. Call LLM to get raw text report
        raw_report = await achat_raw(self.model, prompt)

        # 5. Structure the report using LLM (returns Pydantic model)
        pydantic_report = await prompts.structure_report_with_llm(
            report=raw_report,
            r_squared_dict=self._format_r_squared_dict(sorted_features),
            model=self.model,
            structuring_model=self.structuring_model,
        )
        
        # 6. Convert Pydantic to attrs, enriching with conversation data
        # Extract outcomes as strings for metadata
        outcomes = [str(val) for val in sampled_data.data[problem.target_column.name].to_list()]
        conversations = conversation_strategy.get_conversations(sampled_data, conn)
        conversation_ids = sampled_data.data[conversation_strategy.main_table_id_column.name].to_list()

        # Convert Pydantic to attrs
        report = self._convert_pydantic_to_attrs(
            pydantic_report, sorted_features, tuple(conversations),
            conversation_ids, outcomes, raw_report)

        # Filter out invalid recommendations
        report_with_valid_features = self._filter_recommendations_with_no_supporting_features(report)

        # Filter out invalid conversations
        formatter = self._get_formatter(conversation_strategy, problem, is_batch=False)
        formatted_conversations = await formatter.aformat_batch(sampled_data, conn)
        return await self._verify_conversation_references(report_with_valid_features, formatted_conversations)

