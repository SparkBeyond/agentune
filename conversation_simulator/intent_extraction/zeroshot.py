"""Zero-shot intent extraction implementation using a language model."""

import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional

from ..models.conversation import Conversation
from ..models.intent import Intent
from ..models.roles import ParticipantRole

from .base import IntentExtractor
from .prompts import create_intent_extraction_prompt, format_conversation

logger = logging.getLogger(__name__)


class IntentExtractionResult(BaseModel):
    """Result of intent extraction with reasoning."""
    
    detected: bool = Field(
        description="Whether an intent was detected in the conversation"
    )
    reasoning: str = Field(
        description="Explanation of the extracted intent and why it was chosen"
    )
    role: Optional[ParticipantRole] = Field(
        description="Role of the participant who initiated the intent (CUSTOMER or AGENT), "
                  "or None if no intent was detected"
    )
    description: Optional[str] = Field(
        description="Description of the extracted intent, or None if no intent was detected"
    )
    
    def to_intent(self) -> Intent | None:
        """Convert the extraction result to an Intent object.
        
        Returns:
            An Intent object if an intent was detected with valid role and description, 
            None otherwise
        """
        if not self.detected or self.role is None or self.description is None:
            return None
        return Intent(role=self.role, description=self.description)



class ZeroshotIntentExtractor(IntentExtractor):
    """Zero-shot intent extraction using a language model.
    
    This extractor uses a language model to analyze conversation history and extract
    the primary intent of the conversation initiator (customer or agent).
    """
    
    def __init__(self, llm: BaseChatModel):
        """Initialize the zero-shot intent extractor.
        
        Args:
            llm: Language model to use for intent extraction
        """
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=IntentExtractionResult)
        
        # Create the prompt template with format instructions
        prompt_template = create_intent_extraction_prompt(
            format_instructions=self.parser.get_format_instructions()
        )
        
        # Create the processing chain
        self._chain = prompt_template | self.llm | self.parser
    
    async def extract_intent(self, conversation: Conversation) -> Intent | None:
        """Extract intent from a conversation using a zero-shot approach.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            Extracted intent with reasoning and confidence, or None if extraction fails
        """
        if not conversation.messages:
            return None
            
        try:
            # Format the conversation and process it through the chain
            formatted_conversation = format_conversation(conversation)
            result: IntentExtractionResult = await self._chain.ainvoke({"conversation": formatted_conversation})
            
            return result.to_intent()
            
        except Exception as e:
            # Log the error and return None
            # In a production environment, you might want to log this to a proper logging system
            logger.error(f"Failed to extract intent: {e}")
            return None
