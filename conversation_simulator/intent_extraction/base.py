"""Base class for intent extraction from conversations."""

from __future__ import annotations
import abc

from pydantic import BaseModel, Field

from ..models.conversation import Conversation
from ..models.intent import Intent
from ..models.roles import ParticipantRole


class IntentExtractionResult(BaseModel):
    """Result of intent extraction with reasoning."""
    
    reasoning: str = Field(
        description="Explanation of the extracted intent and why it was chosen"
    )
    role: ParticipantRole = Field(
        description="Role of the participant who initiated the intent (CUSTOMER or AGENT)"
    )
    description: str = Field(
        description="Description of the extracted intent"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the intent extraction (0.0 to 1.0)"
    )
    
    def to_intent(self) -> Intent:
        """Convert the extraction result to an Intent object."""
        return Intent(role=self.role, description=self.description)


class IntentExtractor(abc.ABC):
    """Abstract base class for extracting intents from conversations.
    
    Intent extractors analyze conversation history to determine what participant
    has initiated the conversation and what their intent is.
    """
    
    @abc.abstractmethod
    async def extract_intent(self, conversation: Conversation) -> Intent | None:
        """Extract intent from a conversation.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            Extracted intent or None if no intent could be determined
        """
        ...
