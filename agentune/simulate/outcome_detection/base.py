"""Abstract base class for outcome detection strategies."""

import abc
from typing import cast

from attrs import frozen

from ..models.conversation import Conversation
from ..models.intent import Intent
from ..models.outcome import Outcome, Outcomes

@frozen
class OutcomeDetectionTest:
    conversation: Conversation
    intent: Intent

class OutcomeDetector(abc.ABC):
    """Abstract base class for outcome detection strategies."""
    
    async def detect_outcome(
        self, 
        conversation: Conversation, 
        intent: Intent, 
        possible_outcomes: Outcomes
    ) -> Outcome | None:
        """Detect if conversation has reached any of the possible outcomes.
        
        Args:
            conversation: Current conversation state
            intent: Original intent/goal of the conversation
            possible_outcomes: Set of outcomes to detect
            
        Returns:
            Detected outcome or None if no outcome detected. Underlying errors
            such as connection errors are propagated. Note that None is a valid outcome,
            not an error.
        """
        result = (await self.detect_outcomes(
            (OutcomeDetectionTest(conversation, intent), ),
            possible_outcomes,
            return_exceptions=False
        ))[0]
        return cast(Outcome | None, result)

    @abc.abstractmethod
    async def detect_outcomes(
        self,
        instances: tuple[OutcomeDetectionTest, ...],
        possible_outcomes: Outcomes,
        return_exceptions: bool = True
    ) -> tuple[Outcome | None | Exception, ...]:
        """Detect outcomes for multiple conversations. See detect_outcome for details.
        """
        ...

        
