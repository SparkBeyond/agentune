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


class MockParticipant(Participant):
    """Mock participant for testing that returns predefined messages.
    
    Limitation: Messages should be unique in content and timestamp."""
    
    def __init__(self, role: ParticipantRole, messages: tuple[MessageWithTimestamp, ...]) -> None:
        """Initialize mock participant.
        
        Args:
            role: The role of this participant
            messages: List of messages to return in sequence (None = finished)
        """
        self.role = role
        self.messages = messages

    def with_intent(self, intent_description: str) -> MockParticipant:
        return self # Intent is not used in this mock, so we ignore it

    def _to_message(self, message: MessageWithTimestamp) -> Message:
        return Message(
            content=message.content,
            timestamp=message.timestamp,
            sender=self.role
        )

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Return the next predefined message or None if finished."""
        # Filter conversation messages by our role
        our_messages_content = [msg.content for msg in conversation.messages if msg.sender == self.role]
    
        # iterate over initial_message_from_us and self.messages
        # validate 
        for i, message in enumerate(self.messages):
            if message.content in our_messages_content:
                continue
            # If we reach here, this message has not been added yet
            return self._to_message(message)
        # If we reach here, all messages have been added

        return None


class MockOutcomeDetector(OutcomeDetector):
    """Mock outcome detector for testing."""

    def __init__(self, detect_after_messages: int, outcome: Outcome = Outcome(name="resolved", description="Issue was resolved")) -> None:
        """Initialize mock detector.
        
        Args:
            detect_after_messages: Number of messages after which to detect outcome (None = never)
            outcome: Outcome to return when detected
        """
        self.detect_after_messages = detect_after_messages
        self.outcome = outcome

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
        
        customer = MockParticipant(ParticipantRole.CUSTOMER, customer_messages)
        agent = MockParticipant(ParticipantRole.AGENT, agent_messages)
        
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
        
        customer = MockParticipant(ParticipantRole.CUSTOMER, customer_messages)
        agent = MockParticipant(ParticipantRole.AGENT, agent_messages)
        
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

        customer = MockParticipant(ParticipantRole.CUSTOMER, customer_messages)
        agent = MockParticipant(ParticipantRole.AGENT, agent_messages)
        
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

        customer = MockParticipant(ParticipantRole.CUSTOMER, customer_messages)
        agent = MockParticipant(ParticipantRole.AGENT, agent_messages)
        
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
