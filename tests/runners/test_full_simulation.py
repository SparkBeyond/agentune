"""Tests for FullSimulationRunner."""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import override
import attrs

import pytest

from ._conversation_util import MessageWithTimestamp, MockTurnBasedParticipant, ConversationSplits
from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.intent import Intent
from conversation_simulator.models.message import MessageDraft
from conversation_simulator.models.outcome import Outcome, Outcomes
from conversation_simulator.models.roles import ParticipantRole
from conversation_simulator.outcome_detection.base import OutcomeDetectionTest, OutcomeDetector
from conversation_simulator.runners.full_simulation import FullSimulationRunner


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
    return datetime.now()


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
    async def test_max_messages_limit(
        self,
        base_timestamp: datetime,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        initial_message: MessageDraft
    ) -> None:
        """Test that conversation stops after max_messages is reached."""
        
        # Create mock participants with many messages
        customer_messages = (
            MessageWithTimestamp(
                content="First customer message",
                timestamp=base_timestamp + timedelta(seconds=10),
            ),
            MessageWithTimestamp(
                content="Second customer message",
                timestamp=base_timestamp + timedelta(seconds=30),
            ),
            MessageWithTimestamp(
                content="Third customer message that should not appear",
                timestamp=base_timestamp + timedelta(seconds=50),
            ),
            )
        
        agent_messages = (
            MessageWithTimestamp(
                content="First agent message",
                timestamp=base_timestamp + timedelta(seconds=5),
            ),
            MessageWithTimestamp(
                content="Second agent message",
                timestamp=base_timestamp + timedelta(seconds=20),
            ),
            MessageWithTimestamp(
                content="Third agent message that should not appear",
                timestamp=base_timestamp + timedelta(seconds=40),
            ),
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
            max_messages=3
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
            max_messages_after_outcome=2
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
            max_messages_after_outcome=0  # Immediate termination
        )
        
        # Run simulation
        result = await runner.run()
        
        # Verify immediate termination
        assert result.conversation.outcome is not None
        assert result.conversation.outcome.name == "quick_resolution"
        assert len(result.conversation.messages) == 2  # initial + 1 message that triggered outcome
        assert runner.is_complete

    @pytest.mark.asyncio
    async def test_round_trip_customer_starts_with_passes(
        self,
        conversation_customer_starts_with_passes: Conversation,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        base_timestamp: datetime
    ) -> None:
        """Test round-trip reconstruction of a conversation where customer starts and includes passes."""
        await _assert_round_trip_conversation(
            conversation_customer_starts_with_passes,
            sample_intent,
            sample_outcomes,
            base_timestamp
        )

    @pytest.mark.asyncio
    async def test_round_trip_agent_starts_with_passes(
        self,
        conversation_agent_starts_with_passes: Conversation,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        base_timestamp: datetime
    ) -> None:
        """Test round-trip reconstruction of a conversation where agent starts and includes passes."""
        await _assert_round_trip_conversation(
            conversation_agent_starts_with_passes,
            sample_intent,
            sample_outcomes,
            base_timestamp
        )

    @pytest.mark.asyncio
    async def test_round_trip_simple_back_and_forth(
        self,
        conversation_simple_back_and_forth: Conversation,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        base_timestamp: datetime
    ) -> None:
        """Test round-trip reconstruction of a simple conversation with perfect alternation."""
        await _assert_round_trip_conversation(
            conversation_simple_back_and_forth,
            sample_intent,
            sample_outcomes,
            base_timestamp
        )

    @pytest.mark.asyncio
    async def test_round_trip_multiple_consecutive_passes(
        self,
        conversation_multiple_consecutive_passes: Conversation,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        base_timestamp: datetime
    ) -> None:
        """Test round-trip reconstruction of a conversation with multiple consecutive messages."""
        await _assert_round_trip_conversation(
            conversation_multiple_consecutive_passes,
            sample_intent,
            sample_outcomes,
            base_timestamp
        )

    @pytest.mark.asyncio
    async def test_round_trip_agent_starts_simple(
        self,
        conversation_agent_starts_simple: Conversation,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        base_timestamp: datetime
    ) -> None:
        """Test round-trip reconstruction of a simple agent-initiated conversation."""
        await _assert_round_trip_conversation(
            conversation_agent_starts_simple,
            sample_intent,
            sample_outcomes,
            base_timestamp
        )

    @pytest.mark.asyncio
    async def test_round_trip_single_message(
        self,
        conversation_single_message: Conversation,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        base_timestamp: datetime
    ) -> None:
        """Test round-trip reconstruction of a conversation with just one message."""
        await _assert_round_trip_conversation(
            conversation_single_message,
            sample_intent,
            sample_outcomes,
            base_timestamp
        )

    @pytest.mark.asyncio
    async def test_round_trip_alternating_passes_pattern(
        self,
        conversation_alternating_passes_pattern: Conversation,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        base_timestamp: datetime
    ) -> None:
        """Test round-trip reconstruction of a conversation with complex pass patterns."""
        await _assert_round_trip_conversation(
            conversation_alternating_passes_pattern,
            sample_intent,
            sample_outcomes,
            base_timestamp
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("conversation_fixture_name", [
        "conversation_customer_starts_with_passes",
        "conversation_agent_starts_with_passes", 
        "conversation_simple_back_and_forth",
        "conversation_multiple_consecutive_passes",
        "conversation_agent_starts_simple",
        "conversation_single_message",
        "conversation_alternating_passes_pattern",
    ])
    async def test_round_trip_parametrized(
        self,
        conversation_fixture_name: str,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        base_timestamp: datetime,
        request: pytest.FixtureRequest
    ) -> None:
        """Parametrized test that runs round-trip on all conversation fixtures."""
        # Get the conversation fixture by name
        conversation = request.getfixturevalue(conversation_fixture_name)
        
        await _assert_round_trip_conversation(
            conversation,
            sample_intent,
            sample_outcomes,
            base_timestamp
        )
    

async def _assert_round_trip_conversation(
    conversation: Conversation,
    sample_intent: Intent,
    sample_outcomes: Outcomes,
    base_timestamp: datetime
) -> None:
    """Assert that a conversation can be reconstructed via round-trip through participants.
    
    This helper:
    1. Splits the conversation into participant message sequences
    2. Creates MockTurnBasedParticipants with those sequences
    3. Runs a new simulation using those participants
    4. Verifies the new conversation matches the original exactly
    
    Args:
        conversation: The original conversation to round-trip
        sample_intent: Shared intent fixture to use for simulation
        sample_outcomes: Shared outcomes fixture to use for simulation  
        base_timestamp: Shared base timestamp fixture to use for simulation
    """
    # Split conversation into participant sequences
    splits = ConversationSplits.reconstruct(conversation)
    
    # Create mock participants from the splits
    customer = MockTurnBasedParticipant(ParticipantRole.CUSTOMER, splits.customer_messages)
    agent = MockTurnBasedParticipant(ParticipantRole.AGENT, splits.agent_messages)
    
    # Determine who starts and create initial message
    first_message = conversation.messages[0]
    initial_message = MessageDraft(
        content=first_message.content,
        sender=first_message.sender
    )
    
    # Create runner with shared fixtures
    outcome_detector = MockOutcomeDetector(10000)  # Never detects outcome
    runner = FullSimulationRunner(
        customer=customer,
        agent=agent,
        initial_message=initial_message,
        intent=sample_intent,
        outcomes=sample_outcomes,
        outcome_detector=outcome_detector,
        max_messages=len(conversation.messages) + 5,  # Allow some buffer
    )
    
    # Run the simulation
    result = await runner.run()
    
    # Assert exact match of message contents and senders
    assert len(result.conversation.messages) == len(conversation.messages), (
        f"Message count mismatch: expected {len(conversation.messages)}, "
        f"got {len(result.conversation.messages)}"
    )
    
    for i, (original, reconstructed) in enumerate(zip(conversation.messages, result.conversation.messages)):
        assert original.content == reconstructed.content, (
            f"Message {i} content mismatch: expected '{original.content}', "
            f"got '{reconstructed.content}'"
        )
        assert original.sender == reconstructed.sender, (
            f"Message {i} sender mismatch: expected {original.sender}, "
            f"got {reconstructed.sender}"
        )
