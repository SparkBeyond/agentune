"""Full simulation runner implementation."""

import abc
from datetime import datetime

from attrs import define, field

from .base import Runner
from ..models.conversation import Conversation
from ..models.intent import Intent
from ..models.message import Message, MessageDraft
from ..models.outcome import Outcome, Outcomes
from ..models.results import ConversationResult
from ..models.roles import ParticipantRole
from ..outcome_detection.base import OutcomeDetector
from ..participants.base import Participant


class ProgressHandler(abc.ABC):
    """Abstract base class for handling simulation progress events."""
    
    @abc.abstractmethod
    def on_message_added(self, conversation: Conversation, new_message: Message) -> None:
        """Called when a new message is added to the conversation.
        
        Args:
            conversation: Current conversation state
            new_message: The message that was just added
        """
        ...
    
    @abc.abstractmethod
    def on_outcome_detected(self, conversation: Conversation, outcome: Outcome) -> None:
        """Called when an outcome is detected for the conversation.
        
        Args:
            conversation: Current conversation state
            outcome: The outcome that was detected
        """
        ...
    
    @abc.abstractmethod
    def on_conversation_ended(self, conversation: Conversation, reason: str) -> None:
        """Called when the conversation ends.
        
        Args:
            conversation: Final conversation state
            reason: Reason for ending ("max_messages", "outcome_detected", "participant_finished", etc.)
        """
        ...

@define
class FullSimulationRunner(Runner):
    """Runs conversations with both simulated customer and agent.
    
    Single-use runner that manages conversation state internally.
    Provides progress tracking capabilities for conversations.

    Args:
        customer: Customer participant
        agent: Agent participant
        initial_message: Initial message to start conversation
        intent: Intent for the conversation
        outcomes: Possible outcomes to detect
        outcome_detector: Strategy for detecting conversation outcomes
        max_messages: Maximum number of messages in conversation
        max_messages_after_outcome: Max additional messages after outcome detected (0 = stop immediately)
        base_timestamp: Base timestamp for conversation (current time if None)
    """
    
    customer: Participant
    agent: Participant
    initial_message: MessageDraft
    intent: Intent
    outcomes: Outcomes
    outcome_detector: OutcomeDetector
    max_messages: int = 100
    max_messages_after_outcome: int = 5  # Allow goodbye messages after outcome
    base_timestamp: datetime | None = None  # If None, use current time when run() starts

    # Private state - managed internally
    # Initialize with empty conversation - initial message will be added in run()
    _conversation: Conversation = field(init=False, factory=lambda: Conversation(messages=()))
    _is_complete: bool = field(init=False, default=False)
    _start_time: datetime | None = field(init=False, default=None)
    _outcome_detected: bool = field(init=False, default=False)
    _messages_after_outcome: int = field(init=False, default=0)
    
    async def run(self) -> ConversationResult:
        """Execute the full simulation conversation using a turn-based approach.
        
        The conversation alternates turns between participants. Each participant can
        answer or pass (return None). The conversation ends when both participants
        decide not to answer consecutively.
        
        Returns:
            ConversationResult with conversation history and metadata
        """
        # Initialize timing
        self._start_time = datetime.now()
        current_time = self.base_timestamp or self._start_time
        
        # Add initial message to conversation
        initial_msg = self.initial_message.to_message(current_time)
        self._conversation = self._conversation.add_message(initial_msg)
        
        # Record the role of the initial message to start alternating turns
        current_participant_role = self._alternate_turns(initial_msg.sender)
        
        # Track when a participant passes (returns None) to detect conversation end
        last_was_pass = False
        
        # Main conversation loop - strict alternating turns
        while len(self._conversation.messages) < self.max_messages and not self._is_complete:
            current_participant = self._participant_by_role(current_participant_role)
            
            # Ask the current participant for their next message
            try:
                message = await current_participant.get_next_message(self._conversation)
            except Exception:
                # If the participant had an error, end the conversation
                self._is_complete = True
                break
            
            # Check if the current participant passed (returned None)
            if message is None:
                # If the previous turn was also a pass, end the conversation
                if last_was_pass:
                    self._is_complete = True
                    break
                
                # Mark that this turn was a pass and continue to next participant
                last_was_pass = True
                current_participant_role = self._alternate_turns(current_participant_role)
                continue
            
            # The participant responded with a message, reset the pass tracker
            last_was_pass = False
            
            # Add the message to conversation
            self._conversation = self._conversation.add_message(message)
            
            # Update the current participant role for next turn alternation
            current_participant_role = self._alternate_turns(current_participant_role)
            
            # Check for outcome detection (only if not already detected)
            if not self._outcome_detected:
                detected_outcome = await self.outcome_detector.detect_outcome(
                    self._conversation,
                    self.intent,
                    self.outcomes
                )
                if detected_outcome:
                    self._conversation = self._conversation.set_outcome(detected_outcome)
                    self._outcome_detected = True
                    
                    # If max_messages_after_outcome is 0, end immediately
                    if self.max_messages_after_outcome == 0:
                        self._is_complete = True
                        break
            
            # Track messages after outcome detection
            elif self._outcome_detected:
                self._messages_after_outcome += 1
                if self._messages_after_outcome >= self.max_messages_after_outcome:
                    self._is_complete = True
                    break
        
        # Check if we reached max messages
        if len(self._conversation.messages) >= self.max_messages and not self._is_complete:
            self._is_complete = True
        
        # Calculate duration and return result
        duration = (datetime.now() - self._start_time)
        return ConversationResult(
            conversation=self._conversation,
            duration=duration
        )
    
    @property
    def conversation(self) -> Conversation:
        """Get current conversation state (read-only access)."""
        return self._conversation

    @property
    def is_complete(self) -> bool:
        """Check if the simulation has completed."""
        return self._is_complete

    @staticmethod
    def _alternate_turns(current_participant: ParticipantRole) -> ParticipantRole:
        """Alternate between roles."""
        return ParticipantRole.CUSTOMER if current_participant == ParticipantRole.AGENT else ParticipantRole.AGENT

    def _participant_by_role(self, role: ParticipantRole) -> Participant:
        if role == ParticipantRole.CUSTOMER:
            return self.customer
        elif role == ParticipantRole.AGENT:
            return self.agent
        else:
            raise ValueError(f"Unknown role: {role}")
