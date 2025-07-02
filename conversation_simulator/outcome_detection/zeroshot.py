"""Zero-shot outcome detection implementation."""

from typing import override

from attrs import field, frozen
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from ..models.conversation import Conversation
from ..models.intent import Intent
from ..models.outcome import Outcome, Outcomes
from .base import OutcomeDetectionTest, OutcomeDetector


class OutcomeDetectionResult(BaseModel):
    """Result of outcome detection with reasoning."""
    
    reasoning: str = Field(
        description="Explanation of why this outcome was detected or not detected"
    )
    detected: bool = Field(
        description="Whether a specific outcome was detected in the conversation"
    )
    outcome: str | None = Field(
        default=None,
        description="Name of the detected outcome, or null if no outcome was detected"
    )


@frozen
class ZeroshotOutcomeDetector(OutcomeDetector):
    """Zeroshot outcome detection implementation using a language model.
    
    This detector sends the conversation, intent, and possible outcomes to the
    language model and asks it to determine if any outcome has been reached.
    
    The model returns structured output with reasoning for its decision, enabling
    better transparency and debugging of outcome detection.
    
    Attributes:
        model: LangChain BaseChatModel instance
    """

    model: BaseChatModel = field()
    max_concurrency: int = field(default=50)
    _output_parser = field(init=False, default=PydanticOutputParser(pydantic_object=OutcomeDetectionResult))
    
    _chain: Runnable = field(init=False)
    @_chain.default
    def _create_detection_chain(self) -> Runnable:
        """Create the LangChain chain for outcome detection.
        
        Returns:
            A LangChain chain that processes the input prompts through the model and parser
        """
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("{system_prompt}"),
            HumanMessagePromptTemplate.from_template("{human_prompt}")
        ])
        
        # Build the chain: prompt | model | output_parser
        chain = prompt | self.model | self._output_parser
        
        return chain
    
    def _chain_input(self, instance: OutcomeDetectionTest, possible_outcomes: Outcomes) -> dict[str, str]:
        return {
            "system_prompt": self._build_system_prompt(instance.intent, possible_outcomes),
            "human_prompt": self._build_human_prompt(instance.conversation)
        }

    @override
    async def detect_outcomes(
        self,
        instances: tuple[OutcomeDetectionTest, ...],
        possible_outcomes: Outcomes,
        return_exceptions: bool = True
    ) -> tuple[Outcome | None | Exception, ...]:
        # If conversation is empty, no outcome can be detected and we return None for that index
        valid_indices = [i for i, instance in enumerate(instances) if instance.conversation.messages]
        if not valid_indices:
            return tuple(None for _ in instances)

        indexed_inputs = [(i, self._chain_input(instances[i], possible_outcomes)) for i in valid_indices]
        results = await self._chain.abatch([input for _, input in indexed_inputs], return_exceptions=return_exceptions,
                                     max_concurrency=self.max_concurrency)
        detected_outcome_by_index = {i: result if isinstance(result, Exception) else self._parse_outcome(result, possible_outcomes) 
                                     for (i, _), result in zip(indexed_inputs, results) }
        
        return tuple(detected_outcome_by_index.get(i, None) for i in range(len(instances)))

    
    def _build_system_prompt(self, intent: Intent, possible_outcomes: Outcomes) -> str:
        """Build the system prompt for outcome detection.
        
        Args:
            intent: The conversation intent/goal
            possible_outcomes: Possible outcomes to detect
            
        Returns:
            System prompt string
        """
        outcomes_str = "\n".join([f"- {outcome.name}: {outcome.description}" for outcome in possible_outcomes.outcomes])
        format_instructions = self._output_parser.get_format_instructions()
        
        return f"""You are analyzing a conversation between a customer and an agent to determine if a specific outcome has been reached.

INTENT/GOAL: {intent.role.title()} Intent: {intent.description}

POSSIBLE OUTCOMES:
{outcomes_str}

Your task is to determine if the conversation has reached any of these outcomes or if no outcome has been reached yet.

Provide your analysis with reasoning for your decision. Format your response as a JSON object with the following structure:
{format_instructions}

If an outcome has been reached, set 'detected' to true and specify the exact outcome name in 'outcome'.
If no outcome has been reached yet, set 'detected' to false and 'outcome' to null.
Always provide detailed reasoning for your decision.
"""
    
    def _build_human_prompt(self, conversation: Conversation) -> str:
        """Build the human prompt containing the conversation.
        
        Args:
            conversation: The conversation to analyze
            
        Returns:
            Human prompt string
        """
        conversation_text = "\n".join([
            f"{message.sender}: {message.content}"
            for message in conversation.messages
        ])
        
        return f"""Here is the conversation to analyze:

{conversation_text}

Has this conversation reached one of the defined outcomes? If so, which one?
"""
    
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
