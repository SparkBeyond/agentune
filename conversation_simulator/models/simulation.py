"""Simulation result models for conversation simulation."""

import attrs

from .conversation import Conversation


@attrs.frozen
class ConversationResult:
    """Result of simulating a single conversation."""
    
    conversation: Conversation
    duration_seconds: float = 0.0
    
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
            f"{self.duration_seconds:.2f}s{outcome_str}"
        )
