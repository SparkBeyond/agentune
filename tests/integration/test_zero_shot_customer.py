"""Integration tests for ZeroShotCustomer with real LLM calls."""

import logging
import pytest
from datetime import datetime

from conversation_simulator.participants.customer.zero_shot import ZeroShotCustomer
from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.message import Message
from conversation_simulator.models.roles import ParticipantRole

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestZeroShotCustomerIntegration:
    """Sanity integration tests for ZeroShotCustomer with real LLM."""

    @pytest.mark.asyncio
    async def test_customer_with_intent_responds_to_agent(self, openai_model):
        """Test customer with intent responding to agent message."""
        # Create customer with string intent
        customer = ZeroShotCustomer(
            model=openai_model, 
            intent_description="Find a reliable cloud backup solution for my small business"
        )
        
        # Agent initiates conversation
        agent_message = Message(
            content="Hello! I'm here to help you find the right data protection solution. Can you tell me about your current setup?",
            sender=ParticipantRole.AGENT,
            timestamp=datetime.now()
        )
        
        conversation = Conversation(messages=(agent_message,))
        
        # Customer should respond
        response = await customer.get_next_message(conversation)
        
        assert response is not None
        assert response.sender == ParticipantRole.CUSTOMER
        assert isinstance(response.content, str)
        assert len(response.content.strip()) > 10
        logger.info("Customer response: %s", response.content)

    @pytest.mark.asyncio
    async def test_customer_without_intent_general_response(self, openai_model):
        """Test customer without specific intent."""
        # Create customer without intent
        customer = ZeroShotCustomer(model=openai_model)
        
        # Agent initiates conversation
        agent_message = Message(
            content="Hi there! How can I help you today?",
            sender=ParticipantRole.AGENT,
            timestamp=datetime.now()
        )
        
        conversation = Conversation(messages=(agent_message,))
        
        # Customer should respond
        response = await customer.get_next_message(conversation)
        
        assert response is not None
        assert response.sender == ParticipantRole.CUSTOMER
        assert isinstance(response.content, str)
        assert len(response.content.strip()) > 0
        logger.info("Customer response (no intent): %s", response.content)

    @pytest.mark.asyncio
    async def test_customer_with_intent_method(self, openai_model):
        """Test customer with_intent method."""
        # Create base customer
        base_customer = ZeroShotCustomer(model=openai_model)
        
        # Add intent using with_intent
        customer_with_intent = base_customer.with_intent(
            "I need help setting up automated backups for my photography business"
        )
        
        # Verify the intent was set
        assert customer_with_intent.intent_description == "I need help setting up automated backups for my photography business"
        assert base_customer.intent_description is None  # Original unchanged
        
        # Test response
        agent_message = Message(
            content="What kind of business are you in and what are your data protection needs?",
            sender=ParticipantRole.AGENT,
            timestamp=datetime.now()
        )
        
        conversation = Conversation(messages=(agent_message,))
        response = await customer_with_intent.get_next_message(conversation)
        
        assert response is not None
        assert response.sender == ParticipantRole.CUSTOMER
        assert isinstance(response.content, str)
        assert len(response.content.strip()) > 10
        logger.info("Customer response with intent: %s", response.content)
