"""Base channel interfaces for real agent communication.

⚠️  PLACEHOLDER INTERFACES - NOT WELL DEFINED YET ⚠️

These interfaces are preliminary placeholders for future communication
channel implementations. The actual design and functionality may change
significantly as requirements become clearer.
"""

import abc
from collections.abc import AsyncIterator

from ..models.conversation import Conversation
from ..models.message import Message


class Session(abc.ABC):
    """Abstract base class for conversation sessions.
    
    ⚠️  PLACEHOLDER - Interface not finalized ⚠️
    """
    
    @property
    @abc.abstractmethod
    def session_id(self) -> str:
        """Unique identifier for this session."""
        ...
    
    @abc.abstractmethod
    async def get_conversation(self) -> Conversation:
        """Get the conversation history so far.
        
        Returns:
            Current conversation state
        """
        ...
    
    @abc.abstractmethod
    async def send(self, message: Message) -> None:
        """Send a message to the real agent.
        
        Args:
            message: Message to send
        """
        ...
    
    @abc.abstractmethod
    async def subscribe(self) -> AsyncIterator[Message]:
        """Subscribe to incoming messages from the agent.
        
        Yields:
            Messages received from the real agent
        """
        ...


class Channel(abc.ABC):
    """Abstract base class for communication channels with real agents.
    
    ⚠️  PLACEHOLDER - Interface not finalized ⚠️
    """
    
    @abc.abstractmethod
    async def create_session(self) -> Session:
        """Create a new conversation session.
        
        Returns:
            New session for communication
        """
        ...
