"""Zero-shot intent extraction implementation using a language model."""

from typing import ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from ..models.conversation import Conversation
from .base import IntentExtractionResult, IntentExtractor


class ZeroshotIntentExtractor(IntentExtractor):
    """Zero-shot intent extraction using a language model.
    
    This extractor uses a language model to analyze conversation history and extract
    the primary intent of the conversation initiator (customer or agent).
    """
    
    # Default system prompt for intent extraction
    SYSTEM_PROMPT: ClassVar[str] = """You are an expert at analyzing conversations and identifying user intents.
    
    Your task is to analyze the conversation and determine:
    1. Who was the first to express a clear intent (CUSTOMER or AGENT)
    2. What their specific intent/goal is
    3. Your confidence in this assessment (0.0 to 1.0)
    
    Important notes:
    - The first speaker might just be making a greeting (e.g., "How can I help you?") rather than expressing intent
    - Look for the first message that contains a clear purpose or request
    - The intent could be expressed by either the customer or the agent
    
    Common intent categories include but are not limited to:
    - IT Support (e.g., reporting technical issues, access problems)
    - Sales/Purchasing (e.g., buying a product, inquiring about pricing)
    - Support/Help (e.g., account problems, service issues)
    - Information Request (e.g., asking specific questions, gathering details)
    - Feedback/Complaint (e.g., providing feedback, making a complaint)
    - Scheduling/Booking (e.g., making appointments, reservations)
    
    Be specific in describing the intent. For example, instead of just "IT issue",
    specify the nature of the issue (e.g., "cannot access email account").
    """
    
    # Default human prompt template
    HUMAN_PROMPT_TEMPLATE: ClassVar[str] = """Analyze the following conversation and determine the intent:
    
    {conversation}
    
    Format your response according to the following guidelines:
    {format_instructions}
    
    Your analysis:"""
    
    def __init__(self, llm: BaseChatModel):
        """Initialize the zero-shot intent extractor.
        
        Args:
            llm: Language model to use for intent extraction
        """
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=IntentExtractionResult)
        
        # Set up the prompt templates
        system_prompt = SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT)
        human_prompt = HumanMessagePromptTemplate.from_template(
            self.HUMAN_PROMPT_TEMPLATE,
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = (
            {"conversation": lambda x: "\n".join(
                f"{msg.sender.value.upper()}: {msg.content}" 
                for msg in x.messages
            )}
            | ChatPromptTemplate.from_messages([system_prompt, human_prompt])
            | self.llm
            | self.parser
        )
    
    async def extract_intent(self, conversation: Conversation) -> IntentExtractionResult | None:
        """Extract intent from a conversation using a zero-shot approach.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            Extracted intent with reasoning and confidence, or None if extraction fails
        """
        if not conversation.messages:
            return None
            
        try:
            result = await self.chain.ainvoke(conversation)
            if not isinstance(result, IntentExtractionResult):
                return None
            return result
        except Exception as e:
            # Log the error and return None
            # In a production environment, you might want to log this to a proper logging system
            print(f"Error extracting intent: {e}")
            return None
