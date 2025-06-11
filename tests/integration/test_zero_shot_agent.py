"""Integration tests for ZeroShotAgent with real LLM calls."""

import pytest
from datetime import datetime

from conversation_simulator.participants.agent.zero_shot import ZeroShotAgent
from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.message import Message
from conversation_simulator.models.intent import Intent
from conversation_simulator.models.roles import ParticipantRole


@pytest.mark.integration
class TestZeroShotAgentIntegration:
    """Sanity integration tests for ZeroShotAgent with real LLM."""

    @pytest.mark.asyncio
    async def test_agent_intent_customer_initiates(self, sales_agent_config, openai_model):
        """Test agent with agent-side intent, customer starts conversation."""
        # Agent intent - sales goal
        intent = Intent(
            role=ParticipantRole.AGENT,
            description="Discover if small business owner needs cloud backup solutions"
        )
        
        agent = ZeroShotAgent(sales_agent_config, openai_model, intent)
        
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
        
        # Check timing is realistic (3-20 seconds)
        time_diff = response.timestamp - customer_message.timestamp
        assert 3 <= time_diff.total_seconds() <= 20

    @pytest.mark.asyncio
    async def test_agent_intent_agent_initiates_mentions_intent(self, sales_agent_config, openai_model):
        """Test agent with agent-side intent, agent starts and mentions intent."""
        # Agent intent that should be mentioned
        intent = Intent(
            role=ParticipantRole.AGENT,
            description="Introduce our enterprise cloud backup service to potential business customers"
        )
        
        agent = ZeroShotAgent(sales_agent_config, openai_model, intent)
        
        # Empty conversation - agent initiates
        empty_conversation = Conversation(messages=())
        response = await agent.get_next_message(empty_conversation)
        
        # Basic sanity checks
        assert response is not None
        assert response.sender == ParticipantRole.AGENT
        assert len(response.content.strip()) > 10
        
        # Should mention the service/intent
        response_lower = response.content.lower()
        assert any(word in response_lower for word in ['backup', 'cloud', 'service', 'data', 'enterprise'])

    @pytest.mark.asyncio
    async def test_agent_intent_agent_initiates_no_intent_mention(self, sales_agent_config, openai_model):
        """Test agent with agent-side intent, agent starts but doesn't mention intent directly."""
        # Subtle agent intent
        intent = Intent(
            role=ParticipantRole.AGENT,
            description="Build rapport with potential customer before discussing technology solutions"
        )
        
        agent = ZeroShotAgent(sales_agent_config, openai_model, intent)
        
        # Empty conversation - agent initiates
        empty_conversation = Conversation(messages=())
        response = await agent.get_next_message(empty_conversation)
        
        # Basic sanity checks
        assert response is not None
        assert response.sender == ParticipantRole.AGENT
        assert len(response.content.strip()) > 10
        
        # Should be a greeting/rapport building, not direct sales pitch
        response_lower = response.content.lower()
        assert any(word in response_lower for word in ['hello', 'hi', 'good', 'thank', 'how'])
        
        # This is just a sanity test - we're checking the agent generates reasonable responses
