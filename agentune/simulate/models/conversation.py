"""Conversation models for conversation simulation."""

from __future__ import annotations
import attrs

from .message import Message
from .outcome import Outcome
from .roles import ParticipantRole


@attrs.frozen
class Conversation:
    """Complete conversation with messages and optional outcome."""
    
    messages: tuple[Message, ...]
    outcome: Outcome | None = None
    
    @property
    def customer_messages(self) -> tuple[Message, ...]:
        """Get all messages from the customer."""
        return tuple(msg for msg in self.messages if msg.sender == ParticipantRole.CUSTOMER)
    
    @property
    def agent_messages(self) -> tuple[Message, ...]:
        """Get all messages from agents."""
        return tuple(msg for msg in self.messages if msg.sender == ParticipantRole.AGENT)
    
    @property
    def last_message(self) -> Message | None:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None
    
    @property
    def is_empty(self) -> bool:
        """Check if conversation has no messages."""
        return len(self.messages) == 0

    def add_message(self, message: Message) -> Conversation:
        """Return new conversation with additional message."""
        return attrs.evolve(self, messages=self.messages + (message,))
    
    def set_outcome(self, outcome: Outcome) -> Conversation:
        """Return new conversation with outcome set."""
        return attrs.evolve(self, outcome=outcome)
    
    def __len__(self) -> int:
        """Number of messages in the conversation."""
        return len(self.messages)
    
    def __str__(self) -> str:
        """String representation of the conversation."""
        total = len(self.messages)
        if total == 0:
            return "Conversation (0 messages): <empty>"

        if total <= 20:
            messages_str = "\n".join(str(msg) for msg in self.messages)
        else:
            first = "\n".join(str(msg) for msg in self.messages[:10])
            last = "\n".join(str(msg) for msg in self.messages[-10:])
            messages_str = f"{first}\n...\n{last}"

        outcome_str = f" - Outcome: {self.outcome}" if self.outcome else ""
        return f"Conversation ({total} messages){outcome_str}:\n{messages_str}"
    
    def to_langchain_messages(self) -> list:
        """Convert conversation history to LangChain message format.
        
        Returns:
            List of LangChain BaseMessage objects
        """
        return [msg.to_langchain() for msg in self.messages]
