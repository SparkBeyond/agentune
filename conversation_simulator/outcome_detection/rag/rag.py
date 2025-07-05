"""RAG-based outcome detection implementation.

This module implements outcome detection using Retrieval Augmented Generation (RAG) to find
similar, completed conversations and use them as few-shot examples to guide the language model
in determining whether a conversation has reached a specific outcome.

The implementation follows the same API as ZeroshotOutcomeDetector but enhances accuracy by
leveraging existing conversations from the vector store as examples.
"""


from typing import override
import asyncio

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore

from ...models.conversation import Conversation
from ...models.outcome import Outcome, Outcomes
from ...rag import index_by_prefix
from ..base import OutcomeDetector, OutcomeDetectionTest
from .prompt import OUTCOME_DETECTION_PROMPT_TEMPLATE, format_conversation, format_examples, OutcomeDetectionResult


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
    
    @override
    async def detect_outcomes(
        self, 
        instances: tuple[OutcomeDetectionTest, ...], 
        possible_outcomes: Outcomes,
        return_exceptions: bool = True
    ) -> tuple[Outcome | None | Exception, ...]:
        """Detect if conversations have reached any of the possible outcomes.
        
        This method processes multiple instances in batch. For each instance, it:
        1. Retrieves similar, completed conversations from the vector store
        2. Builds system and human prompts with the retrieved examples
        3. Executes a chain that runs the prompts through the language model
        4. Parses the model's response to determine if an outcome was reached
        
        Args:
            instances: Tuple of test instances containing conversations and intents
            possible_outcomes: Set of defined outcomes to detect
            return_exceptions: Whether to return exceptions or raise them
            
        Returns:
            Tuple of detected outcomes, None values, or exceptions matching the input instances
        """
        # If no instances or all conversations are empty, return all None
        valid_indices = [i for i, instance in enumerate(instances) if instance.conversation.messages]
        if not valid_indices:
            return tuple(None for _ in instances)
        
        # Create the detection chain
        chain = self._create_detection_chain()
        
        # Process each valid instance
        async def prepare_prompt_template_params(instance_idx):
            instance = instances[instance_idx]
            # Retrieve similar conversations for this instance
            few_shot_examples = await self._retrieve_examples(instance.conversation)

            formatted_conversation = format_conversation(instance.conversation)
            formatted_examples = format_examples(few_shot_examples)
            outcomes_str = "\n".join([
                f"- {outcome.name}: {outcome.description}" 
                for outcome in possible_outcomes.outcomes
            ])

            params = {
                "conversation_text": formatted_conversation,
                "examples_text": formatted_examples,
                "intent_role": instance.intent.role.value.capitalize(),
                "intent_description": instance.intent.description,
                "outcomes_str": outcomes_str
            }

            import json
            print(f"""=> 

{json.dumps(params, indent=2)}

""")
            
            # Prepare inputs for the chain
            # 'conversation_text', 'outcomes_str', 'examples_text'
            return instance_idx, params
        
        # Gather all inputs for valid instances
        input_params_by_instance_idx = await asyncio.gather(*[prepare_prompt_template_params(i) for i in valid_indices])
        input_params = [input for _, input in input_params_by_instance_idx]

        # Execute chain in batch mode
        results = await chain.abatch(input_params, return_exceptions=return_exceptions)
        
        # Map results back to original indices
        detected_outcome_by_index: dict[int, Outcome | None | Exception] = {}
        for (idx, _), result in zip(input_params_by_instance_idx, results):
            if isinstance(result, Exception):
                detected_outcome_by_index[idx] = result
            else:
                detected_outcome_by_index[idx] = self._parse_outcome(result, possible_outcomes)
        
        # Return results in original order
        return tuple(detected_outcome_by_index.get(i, None) for i in range(len(instances)))
    
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
        # Build the chain: prompt | model | output_parser
        chain = OUTCOME_DETECTION_PROMPT_TEMPLATE | self.model | self._output_parser
        
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
