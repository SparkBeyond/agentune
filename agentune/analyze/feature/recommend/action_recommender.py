"""Action recommender using LLM to analyze conversation data.

This recommender formats conversation data and uses an LLM to generate
actionable recommendations based on feature importance.
"""

from collections.abc import Sequence
from typing import override

import attrs
import polars as pl
from attrs import frozen
from duckdb import DuckDBPyConnection

from agentune.analyze.core.dataset import Dataset
from agentune.analyze.core.sercontext import LLMWithSpec
from agentune.analyze.feature.gen.insightful_text_generator.features import InsightfulTextFeature
from agentune.analyze.feature.gen.insightful_text_generator.formatting.base import (
    ConversationFormatter,
)
from agentune.analyze.feature.gen.insightful_text_generator.sampling.base import (
    DataSampler,
)
from agentune.analyze.feature.gen.insightful_text_generator.sampling.samplers import (
    BalancedClassSampler,
    BalancedNumericSampler,
)
from agentune.analyze.feature.gen.insightful_text_generator.util import achat_raw
from agentune.analyze.feature.problem import (
    ClassificationProblem,
    Problem,
    RegressionProblem,
)
from agentune.analyze.feature.recommend import prompts
from agentune.analyze.feature.recommend.base import ActionRecommender
from agentune.analyze.feature.stats.base import FeatureWithFullStats
from agentune.analyze.join.conversation import Conversation, ConversationJoinStrategy


@frozen
class FeatureWithScore:
    """Reference to a feature with its importance score."""
    name: str
    sse_reduction: float


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
    """Structured recommendations report with enriched SSE reduction data."""
    analysis_summary: str
    recommendations: tuple[Recommendation, ...]
    conversations: dict[int, ConversationWithMetadata]  # Keyed by display_number (1, 2, 3...)
    raw_report: str


@attrs.frozen
class ConversationActionRecommender(ActionRecommender):
    """Recommender that uses conversation data and LLM to generate actionable recommendations.
    
    This recommender filters conversation features dynamically and generates
    recommendations based on their importance. Currently supports a single
    conversation strategy per recommendation.
    
    The recommender:
    1. Discovers features that use ConversationFormatter
    2. Groups by ConversationJoinStrategy
    3. Validates single strategy (raises error for multiple)
    4. Samples data using problem-appropriate sampler:
       - Classification: BalancedClassSampler (equal samples per class)
       - Regression: BalancedNumericSampler (equal samples per quantile bin)
    5. Formats conversations using ConversationFormatter
    6. Generates LLM-based recommendations
    """
    
    model: LLMWithSpec
    num_samples: int = 40
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
                    num_bins=5,  # Same as feature search default
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
            Tuple of (sorted features by SSE reduction, conversation strategy) if conversation features found.
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
        
        # Sort features by SSE reduction (importance)
        sorted_features = sorted(
            [fws for fws, _ in conv_features],
            key=lambda fws: fws.stats.relationship.sse_reduction,
            reverse=True
        )
        
        return sorted_features, first_strategy

    def _format_sse_reduction_dict(
        self, features_with_stats: list[FeatureWithFullStats]
    ) -> str:
        """Format SSE reduction dictionary as a readable string.
        
        Args:
            features_with_stats: Features sorted by importance
            
        Returns:
            Formatted string showing feature descriptions and SSE reductions
        """
        lines = []
        for i, fws in enumerate(features_with_stats[:self.top_k_features], 1):
            sse_reduction = fws.stats.relationship.sse_reduction
            description = fws.feature.description
            lines.append(f'{i}. {description}: {sse_reduction:.4f}')
        return '\n'.join(lines)

    def _format_conversations(
        self,
        formatted_samples: pl.Series,
    ) -> str:
        """Format conversation samples as a readable string.
        
        Conversations are numbered sequentially (1, 2, 3...) based on their order
        in the formatted_samples series. This ensures the LLM's references
        (e.g., "Conversation 5") can be traced back to the database.
        
        Args:
            formatted_samples: Series of formatted conversation strings
            
        Returns:
            Formatted string with conversations numbered sequentially
        """
        lines = []
        for i, conversation in enumerate(formatted_samples.to_list(), start=1):
            lines.append(f'--- Conversation {i} ---')
            lines.append(str(conversation))
            lines.append('')
        return '\n'.join(lines)

    def _convert_pydantic_to_attrs(
        self,
        pydantic_report: prompts.StructuredReport,
        features_with_stats: list[FeatureWithFullStats],
        conversations: tuple[Conversation | None, ...],
        conversation_ids: list[int | str],
        outcomes: list[str],
        raw_report: str,
    ) -> RecommendationsReport:
        """Convert Pydantic report to attrs report, enriching with SSE reduction and conversation data.
        
        Args:
            pydantic_report: The Pydantic model from LLM structured output
            features_with_stats: Features with their statistics (for SSE lookup)
            conversations: Tuple of Conversation objects (in same order as shown to LLM)
            conversation_ids: List of actual database conversation IDs (in same order as conversations)
            outcomes: List of outcome values (in same order as conversations)
            raw_report: Raw report from LLM structured output
            
        Returns:
            RecommendationsReport with enriched feature references and ConversationWithMetadata objects
        """
        # Build a lookup map: feature description -> SSE reduction
        sse_lookup = {
            fws.feature.description: fws.stats.relationship.sse_reduction
            for fws in features_with_stats
        }
        
        def find_sse_for_feature(feat_name: str) -> float:
            """Find SSE reduction for a feature name.
            
            The LLM is instructed to return exact feature descriptions without SSE values.
            We try exact match first, then fallback to fuzzy matching for robustness.
            """
            # Try exact match first (expected case)
            if feat_name in sse_lookup:
                return sse_lookup[feat_name]
            
            # Fallback: try to find by prefix match (in case LLM didn't follow instructions)
            for desc, sse in sse_lookup.items():
                if feat_name.startswith(desc) or desc.startswith(feat_name):
                    return sse
            
            return 0.0
        
        # Build recommendations and collect all referenced conversation indices
        all_conversation_indices: set[int] = set()
        recommendations_list = []
        
        for rec in pydantic_report.recommendations:
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
                    supporting_features=tuple(
                        FeatureWithScore(
                            name=feat_name,
                            sse_reduction=find_sse_for_feature(feat_name),
                        )
                        for feat_name in rec.supporting_features
                    ),
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
        
        return RecommendationsReport(
            analysis_summary=pydantic_report.analysis_summary,
            recommendations=tuple(recommendations_list),
            conversations=conversations_dict,
            raw_report=raw_report,
        )

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
        sampler = self._get_sampler(problem)
        sampled_data = sampler.sample(dataset, self.num_samples)

        # 3. Fetch conversations once (used for both formatting and final output)
        conversations = conversation_strategy.get_conversations(sampled_data, conn)
        
        # 4. Extract actual conversation IDs from sampled data
        conversation_ids = sampled_data.data[conversation_strategy.main_table_id_column.name].to_list()
        
        # 5. Format conversations using ConversationFormatter
        formatter = ConversationFormatter(
            name='action_recommender_conversations',
            conversation_strategy=conversation_strategy,
            params_to_print=(problem.target_column,),
        )
        formatted_samples = await formatter.aformat_batch(sampled_data, conn)

        # 6. Build prompt (adapts to regression vs classification)
        formatted_conversations = self._format_conversations(formatted_samples)
        if formatter.description is None:
            raise ValueError(f'Formatter {formatter.name} description not available')
        prompt = prompts.create_conversation_analysis_prompt(
            agent_description=self.agent_description,
            instance_description=formatter.description,
            problem=problem,
            sse_reduction_dict=self._format_sse_reduction_dict(sorted_features),
            conversations=formatted_conversations,
        )

        # 7. Call LLM to get raw text report
        raw_report = await achat_raw(self.model, prompt)

        # 8. Structure the report using LLM (returns Pydantic model)
        pydantic_report = await prompts.structure_report_with_llm(
            report=raw_report,
            sse_reduction_dict=self._format_sse_reduction_dict(sorted_features),
            model=self.model,
            structuring_model=self.structuring_model,
        )
        
        # 9. Convert Pydantic to attrs, enriching with SSE reduction and conversation data
        # Extract outcomes as strings for metadata
        outcomes = [str(val) for val in sampled_data.data[problem.target_column.name].to_list()]
        
        return self._convert_pydantic_to_attrs(
            pydantic_report, sorted_features, conversations, 
            conversation_ids, outcomes, raw_report
        )
