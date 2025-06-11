"""Intent models for conversation simulation."""

import attrs

from .roles import ParticipantRole


@attrs.frozen
class Intent:
    """A participant's goal/purpose for the conversation."""
    
    role: ParticipantRole
    description: str
    
    def __str__(self) -> str:
        """String representation of the intent."""
        return f"{self.role.title()} Intent: {self.description}"
