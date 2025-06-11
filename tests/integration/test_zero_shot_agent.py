"""Integration tests for ZeroShotAgent with real LLM calls."""

import logging
import pytest
from datetime import datetime

from conversation_simulator.participants.agent.zero_shot import ZeroShotAgent
from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.message import Message
from conversation_simulator.models.roles import ParticipantRole

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestZeroShotAgentIntegration:
    """Sanity integration tests for ZeroShotAgent with real LLM."""

    @pytest.mark.asyncio
    async def test_agent_intent_customer_initiates(self, sales_agent_config, openai_model):
        """Test agent with agent-side intent, customer starts conversation."""
        # Create agent with string intent
        agent = ZeroShotAgent(
            agent_config=sales_agent_config, 
            model=openai_model, 
            intent_description="Discover if small business owner needs cloud backup solutions"
        )
        
        # Customer initiates conversation
        customer_message = Message(
            content="Hi, we're a small accounting firm looking for better data protection solutions.",
            sender=ParticipantRole.CUSTOMER,
            timestamp=datetime.now()
        )
        
        conversation = Conversation(messages=(customer_message,))
        response = await agent.get_next_message(conversation)
        
        # Basic sanity checks
        assert response is not None
        assert response.sender == ParticipantRole.AGENT
        assert len(response.content.strip()) > 10
        logger.info("Agent response (customer initiated): %s", response.content)
        
        # Check timing is realistic (3-20 seconds)
        time_diff = response.timestamp - customer_message.timestamp
        assert 3 <= time_diff.total_seconds() <= 20

    @pytest.mark.asyncio
    async def test_agent_intent_agent_initiates_mentions_intent(self, sales_agent_config, openai_model):
        """Test agent with agent-side intent, agent starts and mentions intent."""
        # Create agent with intent that should be mentioned
        agent = ZeroShotAgent(
            agent_config=sales_agent_config,
            model=openai_model,
            intent_description="Introduce our enterprise cloud backup service to potential business customers"
        )
        
        # Empty conversation - agent initiates
        empty_conversation = Conversation(messages=())
        response = await agent.get_next_message(empty_conversation)
        
        # Basic sanity checks
        assert response is not None
        assert response.sender == ParticipantRole.AGENT
        assert len(response.content.strip()) > 10
        logger.info("Agent response (agent initiated, with intent): %s", response.content)
        
        # Should mention the service/intent
        response_lower = response.content.lower()
        assert any(word in response_lower for word in ['backup', 'cloud', 'service', 'data', 'enterprise'])

    @pytest.mark.asyncio
    async def test_agent_intent_agent_initiates_no_intent_mention(self, sales_agent_config, openai_model):
        """Test agent with agent-side intent, agent starts but doesn't mention intent directly."""
        # Create agent with subtle intent
        agent = ZeroShotAgent(
            agent_config=sales_agent_config,
            model=openai_model,
            intent_description="Build rapport with potential customer before discussing technology solutions"
        )
        
        # Empty conversation - agent initiates
        empty_conversation = Conversation(messages=())
        response = await agent.get_next_message(empty_conversation)
        
        # Basic sanity checks
        assert response is not None
        assert response.sender == ParticipantRole.AGENT
        assert len(response.content.strip()) > 10
        logger.info("Agent response (rapport building): %s", response.content)
        
        # Should be a greeting/rapport building, not direct sales pitch
        response_lower = response.content.lower()
        assert any(word in response_lower for word in ['hello', 'hi', 'good', 'thank', 'how'])
        
        # This is just a sanity test - we're checking the agent generates reasonable responses

    @pytest.mark.asyncio
    async def test_with_intent_builder_pattern(self, sales_agent_config, openai_model):
        """Test the with_intent builder pattern."""
        # Create agent without intent
        agent = ZeroShotAgent(agent_config=sales_agent_config, model=openai_model)
        assert agent.intent_description is None
        
        # Use builder pattern to create new agent with intent
        intent_description = "Introduce our enterprise cloud backup service to potential business customers"
        agent_with_intent = agent.with_intent(intent_description)
        
        # Verify original agent is unchanged (immutability)
        assert agent.intent_description is None
        assert agent_with_intent.intent_description == intent_description
        assert agent_with_intent is not agent  # Different instances
        
        # Verify new agent works
        empty_conversation = Conversation(messages=())
        response = await agent_with_intent.get_next_message(empty_conversation)
        
        assert response is not None
        assert response.sender == ParticipantRole.AGENT
        assert len(response.content.strip()) > 10
        logger.info("Agent response (with_intent builder pattern): %s", response.content)
