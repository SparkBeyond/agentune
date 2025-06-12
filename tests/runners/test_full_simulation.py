"""Tests for FullSimulationRunner."""

from __future__ import annotations
from datetime import datetime, timedelta
import attrs

import pytest

from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.intent import Intent
from conversation_simulator.models.message import Message, MessageDraft
from conversation_simulator.models.outcome import Outcome, Outcomes
from conversation_simulator.models.roles import ParticipantRole
from conversation_simulator.outcome_detection.base import OutcomeDetector
from conversation_simulator.participants.base import Participant
from conversation_simulator.runners.full_simulation import FullSimulationRunner, ProgressHandler
from conversation_simulator.participants.agent.rag import RagAgent
from conversation_simulator.participants.customer.rag import RagCustomer
from conversation_simulator.rag import create_vector_stores_from_conversations
import os
import json
from pathlib import Path

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

    async def detect_outcome(
        self, 
        conversation: Conversation, 
        intent: Intent, 
        possible_outcomes: Outcomes
    ) -> Outcome | None:
        """Return outcome if conditions are met."""
        if len(conversation.messages) >= self.detect_after_messages:
            return self.outcome
        return None


MOCK_CONVERSATIONS_FOR_RAG_TEST = [
    {
        "id": "rag_conv_1",
        "messages": [
            {"role": "customer", "content": "I need help with my TV."},
            {"role": "agent", "content": "Sure, what is the model number?"},
            {"role": "customer", "content": "It's a Samsung QN90A."},
            {"role": "agent", "content": "Thank you. The QN90A has a known issue with the one connect box. Have you tried unplugging it?"},
        ],
    },
]


class MockProgressHandler(ProgressHandler):
    """Mock progress handler for testing that records all events."""
    
    def __init__(self) -> None:
        """Initialize mock progress handler."""
        self.messages_added: list[tuple[Conversation, Message]] = []
        self.outcomes_detected: list[tuple[Conversation, Outcome]] = []
        self.conversations_ended: list[tuple[Conversation, str]] = []
    
    def on_message_added(self, conversation: Conversation, new_message: Message) -> None:
        """Record message addition."""
        self.messages_added.append((conversation, new_message))
    
    def on_outcome_detected(self, conversation: Conversation, outcome: Outcome) -> None:
        """Record outcome detection."""
        self.outcomes_detected.append((conversation, outcome))
    
    def on_conversation_ended(self, conversation: Conversation, reason: str) -> None:
        """Record conversation ending."""
        self.conversations_ended.append((conversation, reason))


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
        progress_handler = MockProgressHandler()
        
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
            progress_handler=progress_handler
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
        
        # Verify progress handler was called correctly
        assert len(progress_handler.messages_added) == total_expected_messages
        assert len(progress_handler.conversations_ended) == 1
    
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
        progress_handler = MockProgressHandler()
        
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
            progress_handler=progress_handler
        )
        
        # Run simulation
        result = await runner.run()
        
        # Verify conversation stopped at max_messages
        assert len(result.conversation.messages) == 3
        assert runner.is_complete
        assert progress_handler.conversations_ended[-1][1] == "max_messages"
    
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
        progress_handler = MockProgressHandler()
        
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
            progress_handler=progress_handler
        )
        
        # Run simulation
        result = await runner.run()
        
        # Verify outcome was detected and conversation continued for follow-up
        assert result.conversation.outcome is not None
        assert result.conversation.outcome.name == "resolved"
        assert len(result.conversation.messages) == 5  # initial + 2 + 2 follow-up
        assert runner.is_complete
        assert progress_handler.conversations_ended[-1][1] == "outcome_detected_max_followup"
        
        # Verify progress handler recorded outcome detection
        assert len(progress_handler.outcomes_detected) == 1
        assert progress_handler.outcomes_detected[0][1].name == "resolved"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rag_agent_simulation_flow(
        self,
        base_timestamp: datetime,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
    ) -> None:
        """Test a full simulation flow using the RagAgent."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set, skipping RAG integration test.")

        # 1. Create vector stores from historical data
        customer_store, agent_store = await create_vector_stores_from_conversations(
            conversations=MOCK_CONVERSATIONS_FOR_RAG_TEST,
            openai_api_key=openai_api_key,
        )
        assert customer_store is not None
        assert agent_store is not None

        # 2. Setup participants
        # The agent is our RagAgent
        agent = RagAgent(
            agent_vector_store=agent_store,
            customer_vector_store=customer_store,
            all_conversations=MOCK_CONVERSATIONS_FOR_RAG_TEST,
            openai_api_key=openai_api_key,
        )

        # The customer sends one message and then is done.
        customer_messages = (
            MessageWithTimestamp(
                content="My TV is flickering, what should I do?",
                timestamp=base_timestamp + timedelta(seconds=10),
            ),
        )
        customer = MockParticipant(ParticipantRole.CUSTOMER, customer_messages)

        # 3. Setup runner
        initial_message = MessageDraft(
            content="Hello, my TV is broken.",
            sender=ParticipantRole.CUSTOMER
        )
        outcome_detector = MockOutcomeDetector(10000) # Never detects
        progress_handler = MockProgressHandler()

        runner = FullSimulationRunner(
            customer=customer,
            agent=agent,
            initial_message=initial_message,
            intent=sample_intent,
            outcomes=sample_outcomes,
            outcome_detector=outcome_detector,
            progress_handler=progress_handler,
        )

        # 4. Run simulation
        final_conversation = await runner.run()

        # 5. Assert results
        assert final_conversation is not None
        # Initial msg + customer msg + agent response
        assert len(final_conversation.conversation.messages) >= 3
        assert "flickering" in final_conversation.conversation.messages[-1].content.lower()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rag_customer_and_agent_simulation_flow(
        self,
        base_timestamp: datetime,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
    ) -> None:
        """Test a full simulation flow using RagAgent and RagCustomer."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set, skipping RAG integration test.")

        # 1. Create vector stores from historical data
        customer_store, agent_store = await create_vector_stores_from_conversations(
            conversations=MOCK_CONVERSATIONS_FOR_RAG_TEST,
            openai_api_key=openai_api_key,
        )
        assert customer_store is not None
        assert agent_store is not None

        # 2. Setup participants
        agent = RagAgent(
            agent_vector_store=agent_store,
            customer_vector_store=customer_store,
            all_conversations=MOCK_CONVERSATIONS_FOR_RAG_TEST,
            openai_api_key=openai_api_key,
        )
        customer = RagCustomer(
            customer_vector_store=customer_store,
            agent_vector_store=agent_store,
            all_conversations=MOCK_CONVERSATIONS_FOR_RAG_TEST,
            openai_api_key=openai_api_key,
        )

        # 3. Setup runner
        initial_message = MessageDraft(
            content="Hello, my TV is broken.", sender=ParticipantRole.CUSTOMER
        )
        outcome_detector = MockOutcomeDetector(10000)  # Never detects
        progress_handler = MockProgressHandler()

        runner = FullSimulationRunner(
            customer=customer,
            agent=agent,
            initial_message=initial_message,
            intent=sample_intent,
            outcomes=sample_outcomes,
            outcome_detector=outcome_detector,
            progress_handler=progress_handler,
            max_messages=5,  # Limit conversation length for the test
        )

        # 4. Run simulation
        final_conversation = await runner.run()

        # 5. Assert results
        assert final_conversation is not None
        assert len(final_conversation.conversation.messages) >= 3

        # 6. Save conversation log
        log_dir = Path(__file__).parent.parent / "data"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "rag_simulation_log.json"

        conversation_log = []
        for msg in final_conversation.conversation.messages:
            conversation_log.append(
                {
                    "timestamp": msg.timestamp.isoformat(),
                    "sender": msg.sender.value,
                    "content": msg.content,
                }
            )

        with open(log_file_path, "w") as f:
            json.dump(conversation_log, f, indent=2)
        
        print(f"\nConversation log saved to: {log_file_path}")
        
        # Check progress handler
        assert len(progress_handler.messages_added) == len(final_conversation.conversation.messages)
        assert len(progress_handler.conversations_ended) == 1
    
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
        progress_handler = MockProgressHandler()
        
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
            progress_handler=progress_handler
        )
        
        # Run simulation
        result = await runner.run()
        
        # Verify immediate termination
        assert result.conversation.outcome is not None
        assert result.conversation.outcome.name == "quick_resolution"
        assert len(result.conversation.messages) == 2  # initial + 1 message that triggered outcome
        assert runner.is_complete
        assert progress_handler.conversations_ended[-1][1] == "outcome_detected"
