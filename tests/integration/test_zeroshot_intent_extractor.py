"""Integration tests for ZeroshotIntentExtractor with real LLM calls."""

import logging
import pytest
from datetime import datetime

from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.message import Message
from conversation_simulator.models.roles import ParticipantRole
from conversation_simulator.intent_extraction import ZeroshotIntentExtractor

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestZeroshotIntentExtractorIntegration:
    """Integration tests for ZeroshotIntentExtractor with real LLM."""
    
    @pytest.mark.asyncio
    async def test_extract_customer_intent(self, openai_model):
        """Test extraction of customer intent from a conversation."""
        # Create a conversation where customer expresses intent
        now = datetime.now()
        conversation = Conversation(messages=(
            Message(
                sender=ParticipantRole.AGENT, 
                content="Hello! How can I assist you today?", 
                timestamp=now
            ),
            Message(
                sender=ParticipantRole.CUSTOMER, 
                content="I'm having trouble accessing my email account. I keep getting an authentication error.", 
                timestamp=now
            ),
            Message(
                sender=ParticipantRole.AGENT, 
                content="I'm sorry to hear that. Let's try to fix this issue together.", 
                timestamp=now
            ),
        ))
        
        # Create extractor
        extractor = ZeroshotIntentExtractor(llm=openai_model)
        
        # Extract intent
        result = await extractor.extract_intent(conversation)
        
        # Assertions
        assert result is not None
        assert result.role == ParticipantRole.CUSTOMER
        assert "email" in result.description.lower() or "authentication" in result.description.lower()
    
    @pytest.mark.asyncio
    async def test_extract_agent_intent(self, openai_model):
        """Test extraction of agent intent from a conversation."""
        # Create a conversation where agent expresses intent
        now = datetime.now()
        conversation = Conversation(messages=(
            Message(
                sender=ParticipantRole.AGENT, 
                content="Hello! I'm calling because we have a special offer for our loyal customers. "
                       "We can upgrade your current plan to our premium package with 50% off for the first year. "
                       "Would you be interested in hearing more about this exclusive deal?", 
                timestamp=now
            ),
            Message(
                sender=ParticipantRole.CUSTOMER, 
                content="That sounds interesting. What does the premium package include?", 
                timestamp=now
            ),
        ))
        
        # Create extractor
        extractor = ZeroshotIntentExtractor(llm=openai_model)
        
        # Extract intent
        result = await extractor.extract_intent(conversation)
        
        # Assertions
        assert result is not None
        assert result.role == ParticipantRole.AGENT, \
            f"Expected AGENT but got {result.role}. Description: {result.description}"
        assert any(term in result.description.lower() for term in ["upgrade", "premium", "offer", "deal"]), \
            f"Expected intent about upgrade/offer, got: {result.description}"
    
    @pytest.mark.asyncio
    async def test_empty_conversation_returns_none(self, openai_model):
        """Test that an empty conversation returns None."""
        # Create empty conversation
        conversation = Conversation(messages=())
        
        # Create extractor
        extractor = ZeroshotIntentExtractor(llm=openai_model)
        
        # Extract intent
        result = await extractor.extract_intent(conversation)
        
        # Assertions
        assert result is None
    
    @pytest.mark.asyncio
    async def test_extract_intent_with_short_conversation(self, openai_model):
        """Test intent extraction with a very short conversation."""
        # Create a minimal conversation
        now = datetime.now()
        conversation = Conversation(messages=(
            Message(
                sender=ParticipantRole.CUSTOMER, 
                content="I need help with my password reset.", 
                timestamp=now
            ),
        ))
        
        # Create extractor
        extractor = ZeroshotIntentExtractor(llm=openai_model)
        
        # Extract intent
        result = await extractor.extract_intent(conversation)
        
        # Assertions
        assert result is not None
        assert result.role == ParticipantRole.CUSTOMER
        assert "password" in result.description.lower() or "reset" in result.description.lower()
    
    @pytest.mark.asyncio
    async def test_no_intent_detected(self, openai_model):
        """Test that the intent extractor returns None for greeting-only conversations."""
        # Create a conversation with just greetings
        now = datetime.now()
        conversation = Conversation(messages=(
            Message(
                sender=ParticipantRole.AGENT,
                content="Hello! Thank you for calling customer support.",
                timestamp=now
            ),
            Message(
                sender=ParticipantRole.CUSTOMER,
                content="Hi there! How are you today?",
                timestamp=now
            ),
            Message(
                sender=ParticipantRole.AGENT,
                content="I'm doing well, thank you for asking. How can I assist you today?",
                timestamp=now
            ),
        ))
        
        # Create extractor
        extractor = ZeroshotIntentExtractor(llm=openai_model)
        
        # Extract intent
        result = await extractor.extract_intent(conversation)
        
        # Assertions - should return None since there's no clear intent
        assert result is None, "Expected no intent to be detected for greeting-only conversation"
