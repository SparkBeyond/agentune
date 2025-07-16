"""Zero-shot intent extraction implementation using a language model."""

import logging
from attrs import field, frozen
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from typing import override

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
    role: ParticipantRole | None = Field(
        description="Role of the participant who initiated the intent (CUSTOMER or AGENT), "
                  "or None if no intent was detected"
    )
    description: str | None = Field(
        description="Detailed and precise description of the extracted intent, or None if no intent was detected"
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


@frozen
class ZeroshotIntentExtractor(IntentExtractor):
    """Zero-shot intent extraction using a language model.
    
    This extractor uses a language model to analyze conversation history and extract
    the primary intent of the conversation initiator (customer or agent).
    """

    llm: BaseChatModel = field()
    max_concurrency: int = field(default=50)
    parser = field(init=False, default=PydanticOutputParser(pydantic_object=IntentExtractionResult))
    
    _chain: Runnable = field(init=False)
    @_chain.default
    def _chain_default(self) -> Runnable:
        prompt_template = create_intent_extraction_prompt(
            format_instructions=self.parser.get_format_instructions()
        )
        return prompt_template | self.llm | self.parser

    def _chain_input(self, conversation: Conversation) -> dict[str, str]:
        return {"conversation": format_conversation(conversation) }

    @override
    async def extract_intents(self, conversations: tuple[Conversation, ...],
                              return_exceptions: bool = True) -> tuple[Intent | None | Exception, ...]:
        # If conversation is empty, no outcome can be detected and we return None for that index
        valid_indices = [i for i, conversation in enumerate(conversations) if conversation.messages]
        if not valid_indices:
            return tuple(None for _ in conversations)

        indexed_inputs = [(i, self._chain_input(conversations[i])) for i in valid_indices]
        results = await self._chain.abatch([input for _, input in indexed_inputs], return_exceptions=return_exceptions,
                                           max_concurrency=self.max_concurrency)
        detected_outcome_by_index = {i: result.to_intent() for (i, _), result in zip(indexed_inputs, results) }
        
        return tuple(detected_outcome_by_index.get(i, None) for i in range(len(conversations)))
