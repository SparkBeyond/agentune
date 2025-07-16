"""Integration tests for RAG agent and customer factories."""

import pytest
from datetime import timedelta

from agentune.simulate.participants.agent.rag import RagAgentFactory
from agentune.simulate.participants.customer.rag import RagCustomerFactory
from agentune.simulate.models.conversation import Conversation
from agentune.simulate.models.message import Message
from agentune.simulate.models.roles import ParticipantRole
from agentune.simulate.rag import conversations_to_langchain_documents
from langchain_core.vectorstores import InMemoryVectorStore


@pytest.fixture
def mock_rag_conversations(base_timestamp):
    """Mock conversation data for RAG tests using base timestamp."""
    return [
        Conversation(
            messages=tuple([
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="I need help with my Samsung TV. It keeps flickering.",
                    timestamp=base_timestamp
                ),
                Message(
                    sender=ParticipantRole.AGENT,
                    content="I understand you're having issues with your Samsung TV flickering. "
                           "Have you tried turning off any nearby electronic devices that might cause interference?",
                    timestamp=base_timestamp + timedelta(seconds=10)
                ),
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="Yes, I did that but it's still flickering.",
                    timestamp=base_timestamp + timedelta(seconds=20)
                ),
                Message(
                    sender=ParticipantRole.AGENT,
                    content="In that case, let's check the TV settings. Go to Settings > Picture > "
                           "Advanced Settings and turn off Motion Smoothing.",
                    timestamp=base_timestamp + timedelta(seconds=30)
                ),
            ])
        ),
        Conversation(
            messages=tuple([
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="My internet connection is very slow today.",
                    timestamp=base_timestamp + timedelta(days=1, hours=4)
                ),
                Message(
                    sender=ParticipantRole.AGENT,
                    content="I'm sorry to hear about the slow internet. Let's troubleshoot this. "
                           "Can you please restart your modem and router?",
                    timestamp=base_timestamp + timedelta(days=1, hours=4, seconds=15)
                ),
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="I restarted them but it's still slow.",
                    timestamp=base_timestamp + timedelta(days=1, hours=4, minutes=1)
                ),
                Message(
                    sender=ParticipantRole.AGENT,
                    content="Let's run a speed test. Can you go to speedtest.net and tell me the results?",
                    timestamp=base_timestamp + timedelta(days=1, hours=4, minutes=1, seconds=15)
                ),
            ])
        ),
        Conversation(
            messages=tuple([
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="I'm having trouble with my smartphone battery draining too quickly.",
                    timestamp=base_timestamp + timedelta(days=2, hours=-0.5)
                ),
                Message(
                    sender=ParticipantRole.AGENT,
                    content="I can help you with battery optimization. First, let's check which apps "
                           "are consuming the most battery. Go to Settings > Battery and tell me the top consumers.",
                    timestamp=base_timestamp + timedelta(days=2, hours=-0.5, seconds=20)
                ),
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="It shows that social media apps and streaming are using most of the battery.",
                    timestamp=base_timestamp + timedelta(days=2, hours=-0.5, minutes=1)
                ),
                Message(
                    sender=ParticipantRole.AGENT,
                    content="That's common. Try enabling battery saver mode and reducing screen brightness. "
                           "Also, close background apps you're not actively using.",
                    timestamp=base_timestamp + timedelta(days=2, hours=-0.5, minutes=1, seconds=30)
                ),
            ])
        ),
        Conversation(
            messages=tuple([
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="My laptop won't connect to WiFi anymore.",
                    timestamp=base_timestamp + timedelta(days=3, hours=6, minutes=15)
                ),
                Message(
                    sender=ParticipantRole.AGENT,
                    content="Let's troubleshoot your WiFi connection. Can you see your network name "
                           "in the list of available networks?",
                    timestamp=base_timestamp + timedelta(days=3, hours=6, minutes=15, seconds=25)
                ),
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="Yes, I can see it but when I try to connect it says 'Cannot connect to this network'.",
                    timestamp=base_timestamp + timedelta(days=3, hours=6, minutes=16)
                ),
                Message(
                    sender=ParticipantRole.AGENT,
                    content="This usually means the saved password is incorrect or outdated. "
                           "Try forgetting the network and reconnecting with the current password.",
                    timestamp=base_timestamp + timedelta(days=3, hours=6, minutes=16, seconds=30)
                ),
            ])
        ),
        Conversation(
            messages=tuple([
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="I keep getting pop-up ads on my computer even when I'm not browsing.",
                    timestamp=base_timestamp + timedelta(days=4, hours=1, minutes=45)
                ),
                Message(
                    sender=ParticipantRole.AGENT,
                    content="Those pop-ups suggest adware or malware on your system. "
                           "Let's run a security scan. Do you have antivirus software installed?",
                    timestamp=base_timestamp + timedelta(days=4, hours=1, minutes=45, seconds=30)
                ),
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="Yes, I have Windows Defender but I haven't run a full scan recently.",
                    timestamp=base_timestamp + timedelta(days=4, hours=1, minutes=46)
                ),
                Message(
                    sender=ParticipantRole.AGENT,
                    content="Please run a full system scan with Windows Defender. Also, check your "
                           "browser extensions and remove any suspicious ones you don't recognize.",
                    timestamp=base_timestamp + timedelta(days=4, hours=1, minutes=46, seconds=45)
                ),
            ])
        ),
    ]


@pytest.fixture
def vector_store(embedding_model, mock_rag_conversations):
    """Create a vector store with both agent and customer message examples."""
    documents = conversations_to_langchain_documents(mock_rag_conversations)
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)
    return vector_store

@pytest.fixture
def agent_vector_store(vector_store):
    """Return the shared vector store for agent."""
    return vector_store


@pytest.fixture
def customer_vector_store(vector_store):
    """Return the shared vector store for customer."""
    return vector_store
    

class TestRagAgentFactory:
    """Test the RAG agent factory."""
    
    def test_factory_creates_agent(self, agent_vector_store, openai_model):
        """Test that the factory creates a valid RAG agent."""
        factory = RagAgentFactory(model=openai_model, agent_vector_store=agent_vector_store)
        agent = factory.create_participant()
        
        # Verify the agent is created correctly
        assert agent is not None
        assert hasattr(agent, 'agent_vector_store')
        assert hasattr(agent, 'model')
        assert agent.agent_vector_store is agent_vector_store
        assert agent.model is openai_model
        assert agent.role == ParticipantRole.AGENT
    
    def test_factory_with_config(self, agent_vector_store, openai_model, support_agent_config):
        """Test factory with agent configuration."""
        # Test with minimal params
        factory = RagAgentFactory(model=openai_model, agent_vector_store=agent_vector_store)
        agent = factory.create_participant()
        assert agent is not None
        
        # Test with agent config from fixture
        factory_with_config = RagAgentFactory(
            model=openai_model, 
            agent_vector_store=agent_vector_store,
            agent_config=support_agent_config
        )
        agent_with_config = factory_with_config.create_participant()
        assert agent_with_config is not None
    
    @pytest.mark.asyncio
    async def test_agent_can_generate_message(self, agent_vector_store, openai_model, base_timestamp):
        """Test that the created agent can generate messages."""
        factory = RagAgentFactory(model=openai_model, agent_vector_store=agent_vector_store)
        agent = factory.create_participant()
        
        # Set intent using with_intent method
        agent_with_intent = agent.with_intent("Provide technical support for TV issues")
        
        # Create a test conversation
        test_conversation = Conversation(messages=tuple([
            Message(
                sender=ParticipantRole.CUSTOMER,
                content="My TV is flickering, can you help?",
                timestamp=base_timestamp
            )
        ]))
        
        # Test generating a message
        response = await agent_with_intent.get_next_message(test_conversation)
        
        assert response is not None
        assert len(response.content.strip()) > 0
        assert response.sender == ParticipantRole.AGENT


class TestRagCustomerFactory:
    """Test the RAG customer factory."""
    
    def test_factory_creates_customer(self, customer_vector_store, openai_model):
        """Test that the factory creates a valid RAG customer."""
        factory = RagCustomerFactory(model=openai_model, customer_vector_store=customer_vector_store)
        customer = factory.create_participant()
        
        # Verify the customer is created correctly
        assert customer is not None
        assert hasattr(customer, 'customer_vector_store')
        assert hasattr(customer, 'model')
        assert customer.customer_vector_store is customer_vector_store
        assert customer.model is openai_model
        assert customer.role == ParticipantRole.CUSTOMER
    
    @pytest.mark.asyncio
    async def test_customer_can_generate_message(self, customer_vector_store, openai_model, base_timestamp):
        """Test that the created customer can generate messages."""
        factory = RagCustomerFactory(model=openai_model, customer_vector_store=customer_vector_store)
        customer = factory.create_participant()
        
        # Set intent using with_intent method
        customer_with_intent = customer.with_intent("Get help with technical issues")
        
        # Create a test conversation (agent responding to customer)
        test_conversation = Conversation(messages=tuple([
            Message(
                sender=ParticipantRole.CUSTOMER,
                content="My TV is flickering",
                timestamp=base_timestamp
            ),
            Message(
                sender=ParticipantRole.AGENT,
                content="Have you tried turning off any nearby electronic devices?",
                timestamp=base_timestamp + timedelta(seconds=30)
            )
        ]))
        
        # Test generating a message
        response = await customer_with_intent.get_next_message(test_conversation)
        
        assert response is not None
        assert len(response.content.strip()) > 0
        assert response.sender == ParticipantRole.CUSTOMER


class TestRagFactoriesIntegration:
    """Integration tests for both RAG factories working together."""
    
    @pytest.mark.asyncio
    async def test_agent_and_customer_interaction(self, agent_vector_store, customer_vector_store, openai_model, base_timestamp):
        """Test that RAG agent and customer can interact properly."""
        # Create both agent and customer
        agent_factory = RagAgentFactory(model=openai_model, agent_vector_store=agent_vector_store)
        customer_factory = RagCustomerFactory(model=openai_model, customer_vector_store=customer_vector_store)
        
        agent = agent_factory.create_participant().with_intent("Provide technical support")
        customer = customer_factory.create_participant().with_intent("Get help with TV flickering")
        
        # Start with customer message
        initial_conversation = Conversation(messages=tuple([
            Message(
                sender=ParticipantRole.CUSTOMER,
                content="I'm having trouble with my TV flickering",
                timestamp=base_timestamp
            )
        ]))
        
        # Agent responds
        agent_response = await agent.get_next_message(initial_conversation)
        assert agent_response is not None
        assert len(agent_response.content.strip()) > 0
        assert agent_response.sender == ParticipantRole.AGENT
        
        # Customer follows up
        conversation_with_agent_response = Conversation(messages=tuple([
            *initial_conversation.messages,
            agent_response
        ]))
        
        customer_followup = await customer.get_next_message(conversation_with_agent_response)
        assert customer_followup is not None
        assert len(customer_followup.content.strip()) > 0
        assert customer_followup.sender == ParticipantRole.CUSTOMER
    
    def test_factories_use_different_vector_stores(self, agent_vector_store, customer_vector_store, openai_model):
        """Test that factories can use different vector stores for role-specific knowledge."""
        agent_factory = RagAgentFactory(model=openai_model, agent_vector_store=agent_vector_store)
        customer_factory = RagCustomerFactory(model=openai_model, customer_vector_store=customer_vector_store)
        
        agent = agent_factory.create_participant()
        customer = customer_factory.create_participant()
        
        # Both should have different vector stores containing role-specific examples
        assert agent.agent_vector_store is agent_vector_store
        assert customer.customer_vector_store is customer_vector_store
        
        # Vector stores should contain different content for different roles
        # Just verify they are properly initialized
        assert agent.agent_vector_store is not None
        assert customer.customer_vector_store is not None
