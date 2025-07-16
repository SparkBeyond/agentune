"""Analysis models for simulation result evaluation."""

from __future__ import annotations
import attrs

@attrs.frozen
class OutcomeDistribution:
    """Distribution of outcomes in a set of conversations."""
    
    total_conversations: int
    outcome_counts: dict[str, int]  # outcome_name -> count
    conversations_without_outcome: int = 0
    
    @property
    def outcome_percentages(self) -> dict[str, float]:
        """Percentage distribution of outcomes."""
        if self.total_conversations == 0:
            return {}
        return {
            outcome: (count / self.total_conversations) * 100
            for outcome, count in self.outcome_counts.items()
        }
    
    @property
    def no_outcome_percentage(self) -> float:
        """Percentage of conversations without detected outcome."""
        if self.total_conversations == 0:
            return 0.0
        return (self.conversations_without_outcome / self.total_conversations) * 100


@attrs.frozen
class OutcomeDistributionComparison:
    """Comparison between original and simulated outcome distributions."""
    
    original_distribution: OutcomeDistribution
    original_with_predicted_outcomes: OutcomeDistribution
    simulated_distribution: OutcomeDistribution


@attrs.frozen
class MessageDistributionStats:
    """Statistical analysis of message count distributions."""
    
    min_messages: int
    max_messages: int
    mean_messages: float
    median_messages: float
    std_dev_messages: float
    message_count_distribution: dict[int, int]  # message_count -> frequency
    
    @property
    def mode_messages(self) -> int:
        """Most common message count."""
        if not self.message_count_distribution:
            return 0
        return max(self.message_count_distribution.items(), key=lambda x: x[1])[0]


@attrs.frozen
class MessageDistributionComparison:
    """Comparison between original and simulated message count distributions."""
    
    original_stats: MessageDistributionStats
    simulated_stats: MessageDistributionStats


@attrs.frozen
class AdversarialEvaluationResult:
    """Result of adversarial evaluation - trying to distinguish real vs simulated."""
    
    total_pairs_evaluated: int
    correct_identifications: int  # How many times LLM correctly identified real vs simulated
    
    @property
    def accuracy(self) -> float:
        """Accuracy of the adversarial evaluation."""
        if self.total_pairs_evaluated == 0:
            return 0.0
        return self.correct_identifications / self.total_pairs_evaluated
