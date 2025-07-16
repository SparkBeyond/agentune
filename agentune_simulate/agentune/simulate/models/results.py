"""Result models for simulation outcomes."""

from __future__ import annotations
from datetime import datetime, timedelta
import attrs

from .conversation import Conversation
from .scenario import Scenario
from .analysis import OutcomeDistributionComparison, MessageDistributionComparison, AdversarialEvaluationResult


@attrs.frozen
class ConversationResult:
    """Result of simulating a single conversation."""
    
    conversation: Conversation
    duration: timedelta = timedelta(seconds=0)
    
    @property
    def message_count(self) -> int:
        """Number of messages in the conversation."""
        return len(self.conversation.messages)
    
    @property
    def outcome_name(self) -> str | None:
        """Name of the conversation outcome, if any."""
        return self.conversation.outcome.name if self.conversation.outcome else None
    
    def __str__(self) -> str:
        """String representation of the conversation result."""
        outcome_str = f" - {self.outcome_name}" if self.outcome_name else ""
        return (
            f"ConversationResult: {self.message_count} messages, "
            f"{self.duration.total_seconds():.2f}s{outcome_str}"
        )


@attrs.frozen
class OriginalConversation:
    """A real conversation used as input for simulation generation."""
    
    id: str  # Unique identifier for the original conversation
    conversation: Conversation


@attrs.frozen
class SimulatedConversation:
    """A simulated conversation generated from an original conversation."""
    
    id: str  # Unique identifier for this simulated conversation
    scenario_id: str  # ID of the scenario that generated this conversation
    original_conversation_id: str  # Links back to the original
    conversation: Conversation


@attrs.frozen
class SimulationSessionResult:
    """Comprehensive result of a simulation session with analysis capabilities."""
    
    # Session metadata
    session_name: str
    session_description: str
    started_at: datetime
    completed_at: datetime
    
    # Core data
    original_conversations: tuple[OriginalConversation, ...]
    scenarios: tuple[Scenario, ...]
    simulated_conversations: tuple[SimulatedConversation, ...]
    
    # Analysis results
    analysis_result: SimulationAnalysisResult
    
    @property
    def total_original_conversations(self) -> int:
        """Number of original conversations used."""
        return len(self.original_conversations)
    
    @property
    def total_simulated_conversations(self) -> int:
        """Number of simulated conversations generated."""
        return len(self.simulated_conversations)
    
    @property
    def simulation_ratio(self) -> float:
        """Ratio of simulated to original conversations."""
        if self.total_original_conversations == 0:
            return 0.0
        return self.total_simulated_conversations / self.total_original_conversations
    
    def __str__(self) -> str:
        """String representation of the session result."""
        return (
            f"SimulationSessionResult: '{self.session_name}' - "
            f"{self.total_original_conversations} original → {self.total_simulated_conversations} simulated"
        )


@attrs.frozen
class SimulationAnalysisResult:
    """Wrapper for all simulation analysis results."""
    
    outcome_comparison: OutcomeDistributionComparison
    message_distribution_comparison: MessageDistributionComparison
    adversarial_evaluation: AdversarialEvaluationResult
