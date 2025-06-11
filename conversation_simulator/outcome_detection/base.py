"""Abstract base class for outcome detection strategies."""

import abc

from ..models.conversation import Conversation
from ..models.intent import Intent
from ..models.outcome import Outcome, Outcomes


class OutcomeDetector(abc.ABC):
    """Abstract base class for outcome detection strategies."""
    
    @abc.abstractmethod
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
            Detected outcome or None if no outcome detected
        """
        ...
