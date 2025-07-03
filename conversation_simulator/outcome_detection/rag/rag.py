"""RAG-based outcome detection implementation.

This module implements outcome detection using Retrieval Augmented Generation (RAG) to find
similar, completed conversations and use them as few-shot examples to guide the language model
in determining whether a conversation has reached a specific outcome.

The implementation follows the same API as ZeroshotOutcomeDetector but enhances accuracy by
leveraging existing conversations from the vector store as examples.
"""


from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from ...models.conversation import Conversation
from ...models.intent import Intent
from ...models.outcome import Outcome, Outcomes
from ...rag import index_by_prefix
from .base import OutcomeDetector
from .zeroshot import OutcomeDetectionResult
from .prompt import build_system_prompt, build_human_prompt


class RAGOutcomeDetector(OutcomeDetector):
    """RAG-based outcome detection implementation using a language model with retrieval.
    
    This detector retrieves similar conversation examples from a vector store and uses
    them as few-shot examples to help the language model determine if any outcome has
    been reached.
    
    The model returns structured output with reasoning for its decision, enabling
    better transparency and debugging of outcome detection. The implementation is stateless,
    with all prompt logic extracted to module-level functions in the prompt.py file.
    
    Attributes:
        model: LangChain BaseChatModel instance used for outcome detection
        vector_store: Vector store containing indexed conversation examples
        num_examples: Number of few-shot examples to retrieve from the vector store
        _output_parser: Parser for structured output from the language model
    """
    
    def __init__(
        self, 
        model: BaseChatModel, 
        vector_store: VectorStore, 
        num_examples: int = 5
    ):
        """Initialize the detector.
        
        Args:
            model: LangChain BaseChatModel instance to use for detection
            vector_store: Vector store containing conversation examples
            num_examples: Number of few-shot examples to retrieve (default: 5)
        """
        self.model = model
        self.vector_store = vector_store
        self.num_examples = num_examples
        self._output_parser = PydanticOutputParser(pydantic_object=OutcomeDetectionResult)
    
    async def detect_outcome(
        self, 
        conversation: Conversation, 
        intent: Intent, 
        possible_outcomes: Outcomes
    ) -> Outcome | None:
        """Detect if conversation has reached any of the possible outcomes.
        
        This method is stateless and does not modify any instance variables.
        It follows these steps:
        1. Retrieve similar, completed conversations from the vector store
        2. Build system and human prompts with the retrieved examples
        3. Execute a chain that runs the prompts through the language model
        4. Parse the model's response to determine if an outcome was reached
        
        Args:
            conversation: Current conversation state with all messages
            intent: Original intent/goal of the conversation
            possible_outcomes: Set of defined outcomes to detect
            
        Returns:
            Detected outcome object or None if no outcome was detected
        """
        # If conversation is empty, no outcome can be detected
        if not conversation.messages:
            return None
        
        # Retrieve similar conversations from the vector store
        few_shot_examples = await self._retrieve_examples(conversation)
        
        # Set up the detection chain
        chain = self._create_detection_chain()
        
        # Prepare inputs for the chain
        chain_input = {
            "system_prompt": build_system_prompt(self._output_parser),
            "human_prompt": build_human_prompt(
                format_instructions=self._output_parser.get_format_instructions(),
                conversation=conversation,
                examples=few_shot_examples,
                intent=intent,
                possible_outcomes=possible_outcomes
            )
        }
        
        # Execute chain
        result = await chain.ainvoke(chain_input)
        
        # Parse the response to determine if an outcome was detected
        detected_outcome = self._parse_outcome(result, possible_outcomes)
        return detected_outcome
    
    async def _retrieve_examples(self, conversation: Conversation) -> list[tuple[Document, float]]:
        """Retrieve similar conversation examples from the vector store.
        
        Formats the current conversation as a query and searches for similar
        conversations in the vector store. Only completed conversations
        (has_next_message: False) are included to ensure high-quality examples.
        Results are deduplicated by conversation to provide diverse examples.
        
        Args:
            conversation: Current conversation to find examples for
            
        Returns:
            List of similar conversations as (Document, score) tuples, sorted by relevance
            and deduplicated by conversation
        """
        query = index_by_prefix._format_conversation_history(conversation.messages)
        
        # We need to retrieve more examples initially since we'll deduplicate them
        # Aim for 3x our target to ensure we have enough unique conversations
        k_retrieve = self.num_examples * 3
        
        # Retrieve similar conversations, filtering for finished conversations only
        retrieved_docs = await self.vector_store.asimilarity_search_with_score(
            query=query,
            k=k_retrieve,
            filter={"has_next_message": False}
        )
        
        # Sort by similarity score (highest first)
        retrieved_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Deduplicate documents coming from the same conversation by comparing the full_conversation metadata
        unique_docs = []
        seen_conversations = set()
        
        for doc, score in retrieved_docs:
            # Use the full conversation text as a unique identifier
            conversation_text = doc.metadata.get("full_conversation")
            if conversation_text not in seen_conversations:
                unique_docs.append((doc, score))
                seen_conversations.add(conversation_text)
                # Break once we have enough examples
                if len(unique_docs) >= self.num_examples:
                    break
        
        return unique_docs
    
    def _create_detection_chain(self) -> Runnable:
        """Create the LangChain chain for outcome detection.
        
        Returns:
            A LangChain chain that processes the input prompts through the model and parser
        """
        # Get a ChatPromptTemplate with system and human messages
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="{system_prompt}"),
            HumanMessage(content="{human_prompt}")
        ])
        
        # Build the chain: prompt | model | output_parser
        chain = prompt | self.model | self._output_parser
        
        return chain
    
    def _parse_outcome(self, result: OutcomeDetectionResult, possible_outcomes: Outcomes) -> Outcome | None:
        """Parse the detection result to find a matching outcome.
        
        Args:
            result: Structured outcome detection result
            possible_outcomes: Available outcomes to match against
            
        Returns:
            Matched Outcome object or None if no outcome detected
            
        Raises:
            ValueError: If model detected an outcome that doesn't match any defined outcomes
        """
        # If no outcome was detected according to the result
        if not result.detected or not result.outcome:
            return None
            
        # Try to match outcome name
        normalized_outcome = result.outcome.lower().strip()
        
        # Try exact match first
        for outcome in possible_outcomes.outcomes:
            if outcome.name.lower() == normalized_outcome:
                return outcome
                
        # Try partial match if exact match not found
        for outcome in possible_outcomes.outcomes:
            if outcome.name.lower() in normalized_outcome:
                return outcome
                
        # If we get here, the model detected an outcome but we couldn't match it
        # to any of our defined outcomes - this should be treated as an error
        valid_outcomes = ", ".join([o.name for o in possible_outcomes.outcomes])
        raise ValueError(
            f"Model detected outcome '{result.outcome}' which doesn't match any defined outcomes: {valid_outcomes}. "
            f"Reasoning: {result.reasoning}"
        )
