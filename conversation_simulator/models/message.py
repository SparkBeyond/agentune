"""Message models for conversation simulation."""

import attrs
from datetime import datetime

from .roles import ParticipantRole
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@attrs.frozen
class Message:
    """A timestamped message in a conversation."""
    
    content: str
    timestamp: datetime
    sender: ParticipantRole
    
    def __str__(self) -> str:
        """String representation of the message."""
        return f"[{self.sender.upper()}] {self.content}"

    def to_langchain(self) -> BaseMessage:
        """Convert to LangChain message format."""
        if self.sender == ParticipantRole.CUSTOMER:
            return HumanMessage(content=self.content)
        elif self.sender == ParticipantRole.AGENT:
            return AIMessage(content=self.content)
        else:
            # Fallback or error for unknown roles, though ParticipantRole enum should prevent this
            raise ValueError(f"Unknown participant role: {self.sender} for LangChain conversion")


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
