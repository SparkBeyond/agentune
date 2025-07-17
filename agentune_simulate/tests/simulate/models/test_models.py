"""Tests for agentune simulate models."""

from datetime import datetime, timedelta, timezone

from agentune.simulate.models import (
    Message,
    MessageDraft,
    ParticipantRole,
    Intent,
    Outcome,
    Outcomes,
    Conversation,
    ConversationResult,
)


class TestMessage:
    """Test Message model."""
    
    def test_message_creation(self):
        """Test creating a message with all fields."""
        timestamp = datetime.now(timezone.utc)
        message = Message(
            content="Hello world",
            timestamp=timestamp,
            sender=ParticipantRole.CUSTOMER
        )
        
        assert message.content == "Hello world"
        assert message.sender == ParticipantRole.CUSTOMER
        assert message.timestamp == timestamp
    
    def test_message_defaults(self):
        """Test message with default values."""
        timestamp = datetime.now(timezone.utc)
        message = Message(
            content="Test message",
            timestamp=timestamp,
            sender=ParticipantRole.AGENT
        )
        
        assert message.content == "Test message"
        assert message.sender == ParticipantRole.AGENT
        assert message.timestamp == timestamp


class TestMessageDraft:
    """Test MessageDraft model."""
    
    def test_message_draft_creation(self):
        """Test creating a message draft."""
        draft = MessageDraft(
            content="Draft message",
            sender=ParticipantRole.CUSTOMER
        )
        
        assert draft.content == "Draft message"
        assert draft.sender == ParticipantRole.CUSTOMER


class TestIntent:
    """Test Intent model."""
    
    def test_intent_creation(self):
        """Test creating an intent."""
        intent = Intent(
            role=ParticipantRole.CUSTOMER,
            description="Customer requesting assistance"
        )
        
        assert intent.role == ParticipantRole.CUSTOMER
        assert intent.description == "Customer requesting assistance"


class TestOutcome:
    """Test Outcome and OutcomeSchema models."""
    
    def test_outcome_schema_creation(self):
        """Test creating an outcome schema."""
        outcome1 = Outcome(name="resolved", description="Issue resolved successfully")
        outcome2 = Outcome(name="escalated", description="Issue escalated to manager")
        
        schema = Outcomes(
            outcomes=(outcome1, outcome2)
        )
        
        assert len(schema.outcomes) == 2
        assert schema.outcomes[0].name == "resolved"
        assert schema.outcomes[1].name == "escalated"
    
    def test_outcome_creation(self):
        """Test creating an outcome."""
        outcome = Outcome(
            name="success",
            description="Successfully completed"
        )
        
        assert outcome.name == "success"
        assert outcome.description == "Successfully completed"


class TestConversation:
    """Test Conversation model."""
    
    def test_conversation_creation(self):
        """Test creating a conversation."""
        timestamp = datetime.now(timezone.utc)
        messages = (
            Message("Hello", timestamp, ParticipantRole.CUSTOMER),
            Message("Hi there!", timestamp, ParticipantRole.AGENT),
        )
        
        conversation = Conversation(
            messages=messages
        )
        
        assert conversation.messages == messages
        assert conversation.outcome is None
    
    def test_conversation_defaults(self):
        """Test conversation with default values."""
        conversation = Conversation(messages=())
        
        assert conversation.messages == ()
        assert conversation.outcome is None


class TestConversationResult:
    """Test ConversationResult model."""
    
    def test_conversation_result_creation(self):
        """Test creating a conversation result."""
        conversation = Conversation(messages=())
        
        result = ConversationResult(
            conversation=conversation,
            duration=timedelta(seconds=120.5)
        )
        
        assert result.conversation == conversation
        assert result.duration == timedelta(seconds=120.5)
