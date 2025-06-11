"""Message models for conversation simulation."""

import attrs
from datetime import datetime

from .roles import ParticipantRole


@attrs.frozen
class Message:
    """A timestamped message in a conversation."""
    
    content: str
    timestamp: datetime
    sender: ParticipantRole
    
    def __str__(self) -> str:
        """String representation of the message."""
        return f"[{self.sender.upper()}] {self.content}"


@attrs.frozen
class MessageDraft:
    """Message content without timestamp - to be assigned during simulation."""
    
    content: str
    sender: ParticipantRole
    
    def to_message(self, timestamp: datetime) -> Message:
        """Convert to Message with timestamp."""
        return Message(
            content=self.content,
            sender=self.sender,
            timestamp=timestamp
        )
