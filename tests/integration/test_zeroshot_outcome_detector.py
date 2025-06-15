"""Integration tests for ZeroshotOutcomeDetector with real LLM calls."""

import logging
import pytest
from datetime import datetime

from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.intent import Intent
from conversation_simulator.models.message import Message
from conversation_simulator.models.outcome import Outcome, Outcomes
from conversation_simulator.models.roles import ParticipantRole
from conversation_simulator.outcome_detection import ZeroshotOutcomeDetector

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestZeroshotOutcomeDetectorIntegration:
    """Sanity integration tests for ZeroshotOutcomeDetector with real LLM."""
    
    @pytest.mark.asyncio
    async def test_detect_resolved_outcome(self, openai_model):
        """Test detection of a resolved conversation."""
        # Define outcomes
        outcomes = Outcomes(outcomes=(
            Outcome(name="resolved", description="Customer issue was resolved"),
            Outcome(name="escalated", description="Conversation was escalated to supervisor"),
            Outcome(name="pending", description="Issue requires further investigation")
        ))
        
        # Create intent
        intent = Intent(
            role=ParticipantRole.CUSTOMER,
            description="Customer has a billing issue that needs resolution"
        )
        
        # Create conversation with a resolved outcome
        now = datetime.now()  # Use same timestamp for simplicity in tests
        conversation = Conversation(messages=(
            Message(sender=ParticipantRole.CUSTOMER, content="I have a charge on my account I don't recognize.", timestamp=now),
            Message(sender=ParticipantRole.AGENT, content="I understand your concern. Let me look into that for you.", timestamp=now),
            Message(sender=ParticipantRole.AGENT, content="I see the charge is from our subscription renewal on May 1st.", timestamp=now),
            Message(sender=ParticipantRole.CUSTOMER, content="Oh, I forgot about the renewal. Thank you for clarifying.", timestamp=now),
            Message(sender=ParticipantRole.AGENT, content="You're welcome. Is there anything else I can help you with today?", timestamp=now),
            Message(sender=ParticipantRole.CUSTOMER, content="No, that's all. Thanks for your help!", timestamp=now)
        ))
        
        # Create detector
        detector = ZeroshotOutcomeDetector(model=openai_model)
        
        # Detect outcome
        detected_outcome = await detector.detect_outcome(conversation, intent, outcomes)
        
        # Assert outcome was detected correctly
        assert detected_outcome is not None
        assert detected_outcome.name == "resolved"
        logger.info("Detected outcome: %s", detected_outcome)
    
    @pytest.mark.asyncio
    async def test_no_outcome_detected(self, openai_model):
        """Test when no outcome has been reached yet."""
        # Define outcomes
        outcomes = Outcomes(outcomes=(
            Outcome(name="resolved", description="Customer issue was resolved"),
            Outcome(name="escalated", description="Conversation was escalated to supervisor")
        ))
        
        # Create intent
        intent = Intent(
            role=ParticipantRole.CUSTOMER,
            description="Customer has a technical issue with their account"
        )
        
        # Create conversation with no clear outcome yet
        now = datetime.now()  # Use same timestamp for simplicity in tests
        conversation = Conversation(messages=(
            Message(sender=ParticipantRole.CUSTOMER, content="I can't log into my account.", timestamp=now),
            Message(sender=ParticipantRole.AGENT, content="I'm sorry to hear that. Let me help you troubleshoot.", timestamp=now),
            Message(sender=ParticipantRole.AGENT, content="Can you tell me what happens when you try to log in?", timestamp=now),
            Message(sender=ParticipantRole.CUSTOMER, content="It says 'invalid credentials' but I'm sure my password is correct.", timestamp=now)
        ))
        
        # Create detector
        detector = ZeroshotOutcomeDetector(model=openai_model)
        
        # Detect outcome
        detected_outcome = await detector.detect_outcome(conversation, intent, outcomes)
        
        # Assert no outcome was detected
        assert detected_outcome is None
        logger.info("No outcome detected, as expected")
    
    @pytest.mark.asyncio
    async def test_escalated_outcome(self, openai_model):
        """Test detection of an escalated conversation."""
        # Define outcomes
        outcomes = Outcomes(outcomes=(
            Outcome(name="resolved", description="Customer issue was resolved"),
            Outcome(name="escalated", description="Conversation was escalated to supervisor"),
            Outcome(name="pending", description="Issue requires further investigation")
        ))
        
        # Create intent
        intent = Intent(
            role=ParticipantRole.CUSTOMER,
            description="Customer has an account access issue"
        )
        
        # Create conversation with an escalation
        now = datetime.now()  # Use same timestamp for simplicity in tests
        conversation = Conversation(messages=(
            Message(sender=ParticipantRole.CUSTOMER, content="I've been locked out of my account for 3 days!", timestamp=now),
            Message(sender=ParticipantRole.AGENT, content="I apologize for the inconvenience. Let me see what's happening.", timestamp=now),
            Message(sender=ParticipantRole.CUSTOMER, content="This is unacceptable. I need access immediately for a presentation.", timestamp=now),
            Message(sender=ParticipantRole.AGENT, content="I understand the urgency. I need to escalate this to our account security team.", timestamp=now),
            Message(sender=ParticipantRole.CUSTOMER, content="Thank you. How long will that take?", timestamp=now),
            Message(sender=ParticipantRole.AGENT, content="I'll connect you with my supervisor who can expedite this for you.", timestamp=now),
            Message(sender=ParticipantRole.CUSTOMER, content="OK, please hurry.", timestamp=now)
        ))
        
        # Create detector
        detector = ZeroshotOutcomeDetector(model=openai_model)
        
        # Detect outcome
        detected_outcome = await detector.detect_outcome(conversation, intent, outcomes)
        
        # Assert escalation outcome was detected correctly
        assert detected_outcome is not None
        assert detected_outcome.name == "escalated"
        logger.info("Detected outcome: %s", detected_outcome)
    
    @pytest.mark.asyncio
    async def test_empty_conversation(self, openai_model):
        """Test that an empty conversation returns None."""
        # Define outcomes
        outcomes = Outcomes(outcomes=(
            Outcome(name="resolved", description="Customer issue was resolved"),
            Outcome(name="escalated", description="Conversation was escalated to supervisor"),
            Outcome(name="pending", description="Issue requires further investigation")
        ))
        
        # Create intent
        intent = Intent(
            role=ParticipantRole.CUSTOMER,
            description="Customer has a billing question"
        )
        
        # Create empty conversation
        conversation = Conversation(messages=())
        
        # Create detector
        detector = ZeroshotOutcomeDetector(model=openai_model)
        
        # Detect outcome
        detected_outcome = await detector.detect_outcome(conversation, intent, outcomes)
        
        # Assert no outcome is detected for an empty conversation
        assert detected_outcome is None
        logger.info("No outcome detected for empty conversation, as expected")
