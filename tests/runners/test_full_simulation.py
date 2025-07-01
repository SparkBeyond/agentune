"""Tests for FullSimulationRunner."""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import override
import attrs

import pytest

from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.intent import Intent
from conversation_simulator.models.message import Message, MessageDraft
from conversation_simulator.models.outcome import Outcome, Outcomes
from conversation_simulator.models.roles import ParticipantRole
from conversation_simulator.outcome_detection.base import OutcomeDetectionTest, OutcomeDetector
from conversation_simulator.participants.base import Participant
from conversation_simulator.runners.full_simulation import FullSimulationRunner


@attrs.frozen
class MessageWithTimestamp:
    """Message with an associated timestamp."""
    content: str
    timestamp: datetime
    
    def __str__(self) -> str:
        """String representation of the message."""
        return f"{self.timestamp}: {self.content}"


@attrs.frozen
class MockTurnBasedParticipant(Participant):
    """Mock participant for turn-based testing that returns or skips messages in sequence.

    Attributes:
        role: The role of this participant
        messages: List of messages to return in sequence (None = finished)
    """

    role: ParticipantRole
    messages: tuple[MessageWithTimestamp | None, ...]
    message_index = 0

    def with_intent(self, intent_description: str) -> MockTurnBasedParticipant:
        return self  # Intent is not used in this mock

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Return the next message in sequence or None if no more messages or explicitly pass."""
        if self.message_index >= len(self.messages):
            return None

        message = self.messages[self.message_index]
        self.message_index += 1

        # If message is None, this represents a deliberate pass
        if message is None:
            return None


@attrs.frozen
class MockOutcomeDetector(OutcomeDetector):
    """Mock outcome detector for testing.

    Attributes:
        detect_after_messages: Number of messages after which to detect outcome (None = never)
        outcome: Outcome to return when detected
    """

    detect_after_messages: int
    outcome: Outcome = Outcome(name="resolved", description="Issue was resolved")

    @override
    async def detect_outcomes(
        self,
        instances: tuple[OutcomeDetectionTest, ...],
        possible_outcomes: Outcomes,
        return_exceptions: bool = True
    ) -> tuple[Outcome | None | Exception, ...]:
        """Return outcome if conditions are met."""
        return tuple(
            self.outcome if len(instance.conversation.messages) >= self.detect_after_messages else None
            for instance in instances
        )

@pytest.fixture
def base_timestamp() -> datetime:
    """Base timestamp for test messages."""
    return datetime(2024, 1, 1, 10, 0, 0)


@pytest.fixture
def sample_intent() -> Intent:
    """Sample intent for testing."""
    return Intent(
        role=ParticipantRole.CUSTOMER,
        description="Customer wants to inquire about their order"
    )


@pytest.fixture
def sample_outcomes() -> Outcomes:
    """Sample outcomes for testing."""
    return Outcomes(
        outcomes=(
            Outcome(name="resolved", description="Issue was resolved"),
            Outcome(name="escalated", description="Issue was escalated")
        )
    )


@pytest.fixture
def initial_message() -> MessageDraft:
    """Initial message draft for testing."""
    return MessageDraft(
        content="Hello, I need help with my order",
        sender=ParticipantRole.CUSTOMER
    )


class TestFullSimulationRunner:
    """Test cases for FullSimulationRunner."""
    
    @pytest.mark.asyncio
    async def test_basic_conversation_flow(
        self,
        base_timestamp: datetime,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        initial_message: MessageDraft
    ) -> None:
        """Test basic conversation flow with timestamp-based message selection."""
        # Create mock participants with predetermined messages
        customer_messages = (
            MessageWithTimestamp(
                content="Can you check order #12345?",
                timestamp=base_timestamp + timedelta(seconds=10),
            ),
            MessageWithTimestamp(
                content="Thank you for your help!",
                timestamp=base_timestamp + timedelta(seconds=30),
            ),
        )
        
        agent_messages = (
            MessageWithTimestamp(
                content="I'd be happy to help. Let me check that for you.",
                timestamp=base_timestamp + timedelta(seconds=5),  # Earlier timestamp
            ),
            MessageWithTimestamp(
                content="Your order is being processed and will ship tomorrow.",
                timestamp=base_timestamp + timedelta(seconds=20),
            ),
            MessageWithTimestamp(
                content="You're welcome! Is there anything else I can help you with?",
                timestamp=base_timestamp + timedelta(seconds=35),
            ),
        )
        
        customer = MockTurnBasedParticipant(ParticipantRole.CUSTOMER, customer_messages)
        agent = MockTurnBasedParticipant(ParticipantRole.AGENT, agent_messages)
        
        # Create runner
        outcome_detector = MockOutcomeDetector(10000)  # Never detects outcome
        runner = FullSimulationRunner(
            customer=customer,
            agent=agent,
            initial_message=initial_message,
            intent=sample_intent,
            outcomes=sample_outcomes,
            outcome_detector=outcome_detector,
            max_messages=10,
            base_timestamp=base_timestamp,
        )
        
        # Run simulation
        result = await runner.run()
        
        # Verify conversation structure
        total_expected_messages = len(customer_messages) + len(agent_messages) + 1  # initial message + all messages
        assert len(result.conversation.messages) == total_expected_messages
        assert result.conversation.messages[0].content == initial_message.content
        
        # Verify timestamp-based selection (agent message should come first due to earlier timestamp)
        assert result.conversation.messages[1].content == agent_messages[0].content
        assert result.conversation.messages[1].sender == ParticipantRole.AGENT
        
        # Verify next message is customer (next in timestamp order)
        assert result.conversation.messages[2].content == customer_messages[0].content
        assert result.conversation.messages[2].sender == ParticipantRole.CUSTOMER
        
        # Verify conversation ended due to both participants finishing
        assert runner.is_complete
        
    @pytest.mark.asyncio
    async def test_max_messages_limit(
        self,
        base_timestamp: datetime,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        initial_message: MessageDraft
    ) -> None:
        """Test that conversation stops when max_messages is reached."""
        # Create participants that would continue indefinitely
        customer_messages = tuple(
            MessageWithTimestamp(
                content=f"Customer message {i}",
                timestamp=base_timestamp + timedelta(seconds=i*2),
            )
            for i in range(10)
        )
        agent_messages = tuple(
            MessageWithTimestamp(
                content=f"Agent message {i}",
                timestamp=base_timestamp + timedelta(seconds=i*2+1),
                )
            for i in range(10)
        )
        
        customer = MockTurnBasedParticipant(ParticipantRole.CUSTOMER, customer_messages)
        agent = MockTurnBasedParticipant(ParticipantRole.AGENT, agent_messages)
        
        # Create runner with low max_messages
        outcome_detector = MockOutcomeDetector(10000)  # Never detects outcome
        runner = FullSimulationRunner(
            customer=customer,
            agent=agent,
            initial_message=initial_message,
            intent=sample_intent,
            outcomes=sample_outcomes,
            outcome_detector=outcome_detector,
            max_messages=3,  # Low limit
            base_timestamp=base_timestamp,
        )
        
        # Run simulation
        result = await runner.run()
        
        # Verify conversation stopped at max_messages
        assert len(result.conversation.messages) == 3
        assert runner.is_complete
    
    @pytest.mark.asyncio
    async def test_outcome_detection_with_followup_messages(
        self,
        base_timestamp: datetime,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        initial_message: MessageDraft
    ) -> None:
        """Test outcome detection with configurable follow-up messages."""
        
        # Create participants with enough messages
        customer_messages = (
            MessageWithTimestamp(
                content="Problem description",
                timestamp=base_timestamp + timedelta(seconds=5)
            ),
            MessageWithTimestamp(
                content="Thank you!",
                timestamp=base_timestamp + timedelta(seconds=15)            
            ),
            MessageWithTimestamp(
                content="Goodbye!",
                timestamp=base_timestamp + timedelta(seconds=25)
            ),
        )
        agent_messages = (
            MessageWithTimestamp(
                content="I can help",
                timestamp=base_timestamp + timedelta(seconds=10)
            ),
            MessageWithTimestamp(
                content="Here's the solution",
                timestamp=base_timestamp + timedelta(seconds=20)
            ),
            MessageWithTimestamp(
                content="You're welcome!",
                timestamp=base_timestamp + timedelta(seconds=30)
            ),
        )

        customer = MockTurnBasedParticipant(ParticipantRole.CUSTOMER, customer_messages)
        agent = MockTurnBasedParticipant(ParticipantRole.AGENT, agent_messages)
        
        # Test with 2 follow-up messages allowed
        outcome_detector = MockOutcomeDetector(detect_after_messages=3)  # Detect after initial + 2 messages
        runner = FullSimulationRunner(
            customer=customer,
            agent=agent,
            initial_message=initial_message,
            intent=sample_intent,
            outcomes=sample_outcomes,
            outcome_detector=outcome_detector,
            max_messages_after_outcome=2,
            base_timestamp=base_timestamp,
        )
        
        # Run simulation
        result = await runner.run()
        
        # Verify outcome was detected and conversation continued for follow-up
        assert result.conversation.outcome is not None
        assert result.conversation.outcome.name == "resolved"
        assert len(result.conversation.messages) == 5  # initial + 2 + 2 follow-up
        assert runner.is_complete
    
    @pytest.mark.asyncio
    async def test_turn_based_flow(
        self,
        base_timestamp: datetime,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        initial_message: MessageDraft
    ) -> None:
        """Test turn-based conversation flow behavior.

        Specifically validates:
        1. Strict alternating turns between participants
        2. Conversation continuing when one participant passes
        3. Conversation ending when both participants pass consecutively
        """
        # First message from customer (initial), then agent turn, then customer passes,
        # then agent's final message, then both pass to end conversation
        customer_messages = (
            MessageWithTimestamp(
                content="I need help",
                timestamp=base_timestamp + timedelta(seconds=10)
            ),
            None,  # Customer passes their turn
        )
        agent_messages = (
            MessageWithTimestamp(
                content="How can I help?",
                timestamp=base_timestamp + timedelta(seconds=20)
            ),
            MessageWithTimestamp(
                content="Please provide more details",
                timestamp=base_timestamp + timedelta(seconds=30)
            ),
            None,  # Agent passes their turn
        )

        customer = MockTurnBasedParticipant(ParticipantRole.CUSTOMER, customer_messages)
        agent = MockTurnBasedParticipant(ParticipantRole.AGENT, agent_messages)

        runner = FullSimulationRunner(
            customer=customer,
            agent=agent,
            initial_message=initial_message,
            intent=sample_intent,
            outcomes=sample_outcomes,
            outcome_detector=MockOutcomeDetector(detect_after_messages=100),  # Never detect outcome
            base_timestamp=base_timestamp,
        )

        # Run simulation
        result = await runner.run()

        # Verify turn-based behavior
        messages = result.conversation.messages

        # Should have 4 messages: initial + 1 customer + 2 agent
        assert len(messages) == 4

        # Check strict alternating pattern (initial→agent→customer→agent)
        assert messages[0].sender == ParticipantRole.CUSTOMER  # initial
        assert messages[1].sender == ParticipantRole.AGENT     # agent turn
        assert messages[2].sender == ParticipantRole.CUSTOMER  # customer turn
        assert messages[3].sender == ParticipantRole.AGENT     # agent turn

        # Verify message content (confirms turns happened in right order)
        assert messages[1].content == "How can I help?"
        assert messages[2].content == "I need help"
        assert messages[3].content == "Please provide more details"

        # Verify conversation ended due to consecutive passes
        assert runner.is_complete
        # Outcome should be None since we set detect_after_messages to 100
        assert result.conversation.outcome is None

    @pytest.mark.asyncio
    async def test_participant_pass_then_speak_again(
        self,
        base_timestamp: datetime,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        initial_message: MessageDraft
    ) -> None:
        """Test that a participant can pass their turn and then speak on a later turn.

        This illustrates a scenario where:
        1. Customer sends initial message
        2. Agent responds with a greeting
        3. Customer passes their turn (doesn't respond)
        4. Agent sends a reminder message
        5. Customer responds after initially passing
        6. Agent sends a final message
        7. Both participants pass to end the conversation
        """
        # Setup customer message pattern: pass on first turn, then respond, then pass again
        customer_messages = (
            # First turn: pass (explicitly return None)
            None,
            # Second turn: respond after agent's reminder
            MessageWithTimestamp(
                content="Sorry for the delay, I'm here now",
                timestamp=base_timestamp + timedelta(seconds=20),
            ),
        )

        # Setup agent message pattern: greeting, reminder, acknowledgement, then pass
        agent_messages = (
            # First response is a greeting
            MessageWithTimestamp(
                content="Hello, how can I assist you today?",
                timestamp=base_timestamp + timedelta(seconds=5),
            ),
            # Second response is a reminder after customer passes
            MessageWithTimestamp(
                content="Are you still there? I'm waiting to help.",
                timestamp=base_timestamp + timedelta(seconds=15),
            ),
            # Third response acknowledges customer's return
            MessageWithTimestamp(
                content="Thanks for returning! How can I help?",
                timestamp=base_timestamp + timedelta(seconds=25),
            ),
        )

        # Create participants using the MockTurnBasedParticipant class which supports explicit passing
        customer = MockTurnBasedParticipant(ParticipantRole.CUSTOMER, customer_messages)
        agent = MockTurnBasedParticipant(ParticipantRole.AGENT, agent_messages)

        # Create outcome detector that never detects an outcome
        outcome_detector = MockOutcomeDetector(100)  # Only detects after 100 messages

        # Create runner
        runner = FullSimulationRunner(
            customer=customer,
            agent=agent,
            initial_message=initial_message,
            intent=sample_intent,
            outcomes=sample_outcomes,
            outcome_detector=outcome_detector,
            base_timestamp=base_timestamp,
        )

        # Run simulation
        result = await runner.run()

        # Get messages for easier assertions
        messages = result.conversation.messages

        # Verify message sequence with customer passing and then speaking again
        assert len(messages) == 5  # initial + 4 more messages
        assert messages[0].sender == ParticipantRole.CUSTOMER  # initial message
        assert messages[1].sender == ParticipantRole.AGENT     # greeting
        assert messages[2].sender == ParticipantRole.AGENT     # reminder (customer passed)
        assert messages[3].sender == ParticipantRole.CUSTOMER  # customer returns
        assert messages[4].sender == ParticipantRole.AGENT     # agent acknowledgement

        # Verify message content
        assert messages[1].content == "Hello, how can I assist you today?"
        assert messages[2].content == "Are you still there? I'm waiting to help."
        assert messages[3].content == "Sorry for the delay, I'm here now"
        assert messages[4].content == "Thanks for returning! How can I help?"

        # Verify conversation ended due to consecutive passes
        assert runner.is_complete
        assert result.conversation.outcome is None

    @pytest.mark.asyncio
    async def test_immediate_termination_on_outcome(
        self,
        base_timestamp: datetime,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        initial_message: MessageDraft
    ) -> None:
        """Test immediate termination when max_messages_after_outcome is 0."""
        
        customer_messages = (
            MessageWithTimestamp(
                content="Quick question",
                timestamp=base_timestamp + timedelta(seconds=5)
            ),
            MessageWithTimestamp(
                content="Should not appear",
                timestamp=base_timestamp + timedelta(seconds=15)
            ),
        )
        agent_messages = (
            MessageWithTimestamp(
                content="Quick answer",
                timestamp=base_timestamp + timedelta(seconds=10)
            ),
            MessageWithTimestamp(
                content="Should not appear",
                timestamp=base_timestamp + timedelta(seconds=20)
            ),
        )

        customer = MockTurnBasedParticipant(ParticipantRole.CUSTOMER, customer_messages)
        agent = MockTurnBasedParticipant(ParticipantRole.AGENT, agent_messages)
        
        # Test with immediate termination
        outcome_detector = MockOutcomeDetector(detect_after_messages=2, outcome=Outcome(name="quick_resolution", description="Quick resolution achieved"))  # Detect after initial + 1 message
        runner = FullSimulationRunner(
            customer=customer,
            agent=agent,
            initial_message=initial_message,
            intent=sample_intent,
            outcomes=sample_outcomes,
            outcome_detector=outcome_detector,
            max_messages_after_outcome=0,  # Immediate termination
            base_timestamp=base_timestamp,
        )
        
        # Run simulation
        result = await runner.run()
        
        # Verify immediate termination
        assert result.conversation.outcome is not None
        assert result.conversation.outcome.name == "quick_resolution"
        assert len(result.conversation.messages) == 2  # initial + 1 message that triggered outcome
        assert runner.is_complete
