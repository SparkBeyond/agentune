"""Integration tests for RAG agent functionality."""

import os
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
import logging

from agentune.simulate.participants.agent.rag import RagAgent
from agentune.simulate.models.conversation import Conversation
from agentune.simulate.models.message import Message, MessageDraft
from agentune.simulate.models.roles import ParticipantRole
from agentune.simulate.models.intent import Intent
from agentune.simulate.models.outcome import Outcome, Outcomes
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from agentune.simulate.rag import conversations_to_langchain_documents


from agentune.simulate.runners.full_simulation import FullSimulationRunner
from tests.simulate.runners.test_full_simulation import (
    MockTurnBasedParticipant, 
    MessageWithTimestamp, 
    MockOutcomeDetector 
)

logger = logging.getLogger(__name__)

# Mock conversation data for RAG tests
MOCK_RAG_CONVERSATIONS = [
    Conversation(
        messages=tuple([
            Message(
                sender=ParticipantRole.CUSTOMER, 
                content="I need help with my Samsung TV. It keeps flickering.", 
                timestamp=datetime(2023, 5, 1, 10, 0, 0)
            ),
            Message(
                sender=ParticipantRole.AGENT, 
                content="I understand you're having issues with your Samsung TV flickering. Have you tried turning off any nearby electronic devices that might cause interference?", 
                timestamp=datetime(2023, 5, 1, 10, 0, 10)
            ),
            Message(
                sender=ParticipantRole.CUSTOMER, 
                content="Yes, I did that but it's still flickering.", 
                timestamp=datetime(2023, 5, 1, 10, 0, 20)
            ),
            Message(
                sender=ParticipantRole.AGENT, 
                content="Let's try a power cycle. Unplug your TV from the wall for about 30 seconds, then plug it back in. This often resolves flickering issues with Samsung TVs.", 
                timestamp=datetime(2023, 5, 1, 10, 0, 30)
            ),
        ])
    ),
    Conversation(
        messages=tuple([
            Message(
                sender=ParticipantRole.CUSTOMER, 
                content="My internet connection drops frequently.", 
                timestamp=datetime(2023, 5, 2, 14, 0, 0)
            ),
            Message(
                sender=ParticipantRole.AGENT, 
                content="I'm sorry to hear about your internet connection issues. How often does it disconnect and have you noticed any patterns?", 
                timestamp=datetime(2023, 5, 2, 14, 0, 10)
            ),
            Message(
                sender=ParticipantRole.CUSTOMER, 
                content="It happens every hour or so, especially in the evenings.", 
                timestamp=datetime(2023, 5, 2, 14, 0, 20)
            ),
            Message(
                sender=ParticipantRole.AGENT, 
                content="Evening disconnections often suggest network congestion. Try changing your router's WiFi channel in the settings to avoid interference from neighbors' networks.", 
                timestamp=datetime(2023, 5, 2, 14, 0, 30)
            ),
        ])
    ),
]


@pytest.mark.integration
class TestRagAgentIntegration:
    """Integration tests for RAG agent with real vector stores and LLM."""
    
    @pytest_asyncio.fixture(scope="class")
    async def agent_vector_store(self, request, embedding_model):
        """Create in-memory vector stores for testing.
        
        This fixture ensures no disk operations are performed during testing.
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set, skipping RAG integration test.")
        
        # Create vector stores directly in memory
        agent_documents = conversations_to_langchain_documents(MOCK_RAG_CONVERSATIONS)
        # Filter documents where next_message_role is AGENT
        agent_documents = [doc for doc in agent_documents if doc.metadata.get('next_message_role') == ParticipantRole.AGENT.value]
        agent_store = InMemoryVectorStore(embedding=embedding_model)
        await agent_store.aadd_documents(agent_documents)        
        assert isinstance(agent_store, VectorStore)
        return agent_store

    @pytest.fixture(scope="class")
    def base_timestamp(self) -> datetime:
        return datetime.now()

    @pytest.fixture(scope="class")
    def sample_intent(self) -> Intent:
        return Intent(role=ParticipantRole.CUSTOMER, description="Resolve TV flickering issue")

    @pytest.fixture(scope="class")
    def sample_outcomes(self) -> Outcomes:
        return Outcomes(
            outcomes=(
                Outcome(name="resolved", description="Issue was resolved."),
                Outcome(name="not_resolved", description="Issue was not resolved."),
            )
        )

    @pytest.mark.asyncio
    async def test_rag_agent_responds_to_related_query(self, agent_vector_store, openai_model):
        """Test RagAgent responds appropriately to a query related to vector store content."""
        
        # Create RAG agent
        agent = RagAgent(agent_vector_store=agent_vector_store, model=openai_model)
        
        # Create a conversation with a related query
        customer_message = Message(
            content="My Samsung TV screen is flickering on and off. Can you help?",
            sender=ParticipantRole.CUSTOMER,
            timestamp=datetime.now()
        )
        
        conversation = Conversation(messages=(customer_message,))
        response = await agent.get_next_message(conversation)
        
        # Assertions
        assert response is not None
        assert response.sender == ParticipantRole.AGENT
        assert len(response.content) > 20
        
        # Basic response validation
        assert len(response.content) > 20, "Response should be more than 20 characters"
        logger.info("RAG agent response (TV query): %s", response.content)
        
        # Check that the response is relevant to the query
        assert any(phrase in response.content.lower() for phrase in ["tv", "television"]), \
            "Response should be related to TV issues"

    @pytest.mark.asyncio
    async def test_rag_agent_responds_to_unrelated_query(self, agent_vector_store, openai_model):
        """Test RagAgent can respond to a query unrelated to vector store content."""
        
        # Create RAG agent
        agent = RagAgent(agent_vector_store=agent_vector_store, model=openai_model)
        
        # Create a conversation with an unrelated query
        customer_message = Message(
            content="I'm looking for information about your store's return policy.",
            sender=ParticipantRole.CUSTOMER,
            timestamp=datetime.now()
        )
        
        conversation = Conversation(messages=(customer_message,))
        response = await agent.get_next_message(conversation)
        
        # Assertions
        assert response is not None
        assert response.sender == ParticipantRole.AGENT
        assert len(response.content) > 20
        
        # Log response for debugging
        logger.info("RAG agent response (unrelated query): %s", response.content)
        
        # For an unrelated query, RAG should ideally not pull in TV-specific or internet-specific details
        response_lower = response.content.lower()
        assert "flicker" not in response_lower, "Response to unrelated query should not contain 'flicker' from TV RAG data."
        assert "samsung" not in response_lower, "Response to unrelated query should not contain 'samsung' from TV RAG data."
        assert "power cycle" not in response_lower, "Response to unrelated query should not contain 'power cycle' from TV RAG data."
        assert "internet" not in response_lower, "Response to unrelated query should not contain 'internet' from other RAG data."
        assert "wifi" not in response_lower, "Response to unrelated query should not contain 'wifi' from other RAG data."

        # It should, however, attempt to answer the question about return policy
        assert "return" in response_lower or "policy" in response_lower, \
            f"Response should address the return policy query. Got: {response.content}"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rag_agent_simulation_flow(
        self,
        base_timestamp: datetime,
        agent_vector_store: VectorStore,
        sample_intent: Intent,
        sample_outcomes: Outcomes,
        openai_model
    ) -> None:
        """Test a full simulation flow using the RagAgent."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set, skipping RAG integration test.")

        # 1. Create vector stores from historical data
        agent_store = agent_vector_store
        assert agent_store is not None

        # 2. Setup participants
        # The agent is our RagAgent
        agent = RagAgent(agent_vector_store=agent_store, model=openai_model)
        intent_description = "Provide TV troubleshooting solutions"
        agent = agent.with_intent(intent_description)

        # The customer sends one message and then is done.
        customer_messages = (
            MessageWithTimestamp(
                content="My TV is flickering, what should I do?",
                timestamp=base_timestamp + timedelta(seconds=10),
            ),
            None
        )
        customer = MockTurnBasedParticipant(ParticipantRole.CUSTOMER, customer_messages)

        # 3. Setup runner
        initial_message = MessageDraft(
            content="Hello, my TV is broken.",
            sender=ParticipantRole.CUSTOMER
        )
        outcome_detector = MockOutcomeDetector(10000)  # Never detects

        runner = FullSimulationRunner(
            customer=customer,
            agent=agent,
            initial_message=initial_message,
            intent=sample_intent,
            outcomes=sample_outcomes,
            outcome_detector=outcome_detector,
        )

        # 4. Run simulation
        final_conversation_result = await runner.run()

        # 5. Assert results
        assert final_conversation_result is not None
        # Initial msg + customer msg + agent response
        assert len(final_conversation_result.conversation.messages) >= 3
        # Check if the agent's response (last message) contains relevant term
        last_message_content = final_conversation_result.conversation.messages[-1].content.lower()
        assert "flicker" in last_message_content or "samsung" in last_message_content or "tv" in last_message_content, \
            f"Agent response should be relevant to TV flickering. Got: {final_conversation_result.conversation.messages[-1].content}"
