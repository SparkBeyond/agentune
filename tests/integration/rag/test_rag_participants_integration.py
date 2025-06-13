"""Integration tests for RAG participants with real LLM calls."""

import os
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
import logging

from conversation_simulator.participants.agent.rag import RagAgent
from conversation_simulator.participants.customer.rag import RagCustomer
from conversation_simulator.rag import create_vector_stores_from_conversations
from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.message import Message
from conversation_simulator.models.roles import ParticipantRole
from langchain_core.vectorstores import VectorStore
from pathlib import Path
import json
from langchain_community.vectorstores import FAISS

from conversation_simulator.runners.full_simulation import FullSimulationRunner
from conversation_simulator.models.intent import Intent
from conversation_simulator.models.outcome import Outcome, Outcomes
from conversation_simulator.models.message import MessageDraft
from tests.runners.test_full_simulation import (
    MockParticipant, 
    MessageWithTimestamp, 
    MockOutcomeDetector, 
    MockProgressHandler
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

# Define MOCK_CONVERSATIONS_FOR_RAG_TEST (can be same as MOCK_RAG_CONVERSATIONS or different)
# For this example, we'll use the same data for simplicity.
MOCK_CONVERSATIONS_FOR_RAG_TEST = MOCK_RAG_CONVERSATIONS


@pytest.mark.integration
class TestRagParticipantsIntegration:
    """Integration tests for RAG participants with real vector stores and LLM."""
    
    @pytest_asyncio.fixture(scope="class")
    async def vector_stores(self, request):
        """Create in-memory vector stores for testing.
        
        This fixture ensures no disk operations are performed during testing.
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            pytest.skip("OPENAI_API_KEY not set, skipping RAG integration test.")
        

        # Create vector stores directly in memory
        customer_store, agent_store = await create_vector_stores_from_conversations(
            conversations=MOCK_RAG_CONVERSATIONS,
            vector_store_class=FAISS,
            openai_embedding_model_name="text-embedding-ada-002"
        )
        
        assert isinstance(customer_store, VectorStore)
        assert isinstance(agent_store, VectorStore)
        
        # Add cleanup to ensure no files are left behind
        def cleanup():
            # FAISS in-memory stores don't need cleanup, but this is a safeguard
            if hasattr(customer_store, 'delete'):
                try:
                    customer_store.delete()
                except Exception as e:
                    logger.warning(f"Error cleaning up customer store: {e}")
            if hasattr(agent_store, 'delete'):
                try:
                    agent_store.delete()
                except Exception as e:
                    logger.warning(f"Error cleaning up agent store: {e}")
        
        request.addfinalizer(cleanup)
        return customer_store, agent_store

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
    async def test_rag_agent_responds_to_related_query(self, vector_stores):
        """Test RagAgent responds appropriately to a query related to vector store content."""
        customer_store, agent_store = vector_stores
        
        # Create RAG agent
        agent = RagAgent(agent_vector_store=agent_store)
        
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
    async def test_rag_agent_responds_to_unrelated_query(self, vector_stores):
        """Test RagAgent can respond to a query unrelated to vector store content."""
        customer_store, agent_store = vector_stores
        
        # Create RAG agent
        agent = RagAgent(agent_vector_store=agent_store)
        
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
        # from MOCK_RAG_CONVERSATIONS. The response should be more generic or try to address the actual query.
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
    async def test_rag_customer_responds_to_agent_query(self, vector_stores):
        """Test RagCustomer responds appropriately to an agent query."""
        customer_store, _ = vector_stores
        
        # Create RAG customer
        customer = RagCustomer(customer_vector_store=customer_store)
        
        # Create a conversation with an agent query
        agent_message = Message(
            content="I see you're having issues with your TV. Can you tell me what model it is and what specific problems you're experiencing?",
            sender=ParticipantRole.AGENT,
            timestamp=datetime.now()
        )
        
        conversation = Conversation(messages=(agent_message,))
        response = await customer.get_next_message(conversation)
        
        # Assertions
        assert response is not None
        assert response.sender == ParticipantRole.CUSTOMER
        assert len(response.content) > 20
        
        # Log response for debugging
        logger.info("RAG customer response: %s", response.content)
        
        # Basic response validation
        assert len(response.content) > 20, "Response should be more than 20 characters"
        logger.info("RAG customer response: %s", response.content)
        
        # Check that the response is relevant to TV issues (case-insensitive check)
        response_lower = response.content.lower()
        assert any(phrase in response_lower for phrase in ["tv", "television", "samsung", "qled"]), \
            f"Response should be related to TV issues. Response was: {response.content}"

    @pytest.mark.asyncio
    async def test_rag_multi_turn_conversation(self, vector_stores):
        """Test a multi-turn conversation between RagAgent and RagCustomer."""
        customer_store, agent_store = vector_stores
        
        # Create RAG participants
        agent = RagAgent(agent_vector_store=agent_store)
        customer = RagCustomer(customer_vector_store=customer_store)
        
        # Initialize conversation
        now = datetime.now()
        messages = [
            Message(
                content="I'm having problems with my Samsung TV.",
                sender=ParticipantRole.CUSTOMER,
                timestamp=now
            )
        ]
        conversation = Conversation(messages=tuple(messages))
        
        # First agent response
        agent_response = await agent.get_next_message(conversation)
        assert agent_response is not None
        messages.append(agent_response)
        conversation = Conversation(messages=tuple(messages))
        
        # Customer reply
        customer_response = await customer.get_next_message(conversation)
        assert customer_response is not None
        messages.append(customer_response)
        conversation = Conversation(messages=tuple(messages))
        
        # Second agent response
        agent_response2 = await agent.get_next_message(conversation)
        assert agent_response2 is not None
        messages.append(agent_response2)
        
        # Log the conversation
        logger.info("Multi-turn conversation:")
        for msg in messages:
            logger.info(f"{msg.sender.value}: {msg.content}")
            
        # Assertions about the conversation
        assert len(messages) == 4
        
        # Check if relevant terms appear anywhere in the conversation
        all_content = " ".join([msg.content.lower() for msg in messages])
        logger.info("Full conversation content: %s", all_content)
        
        tv_terms = ["samsung", "tv", "television", "screen"]
        issue_terms = ["flicker", "problem", "issue", "broken"]
        solution_terms = ["try", "recommend", "suggest", "unplug", "power", "reset", "help"]
        
        tv_matches = [term for term in tv_terms if term in all_content]
        issue_matches = [term for term in issue_terms if term in all_content]
        solution_matches = [term for term in solution_terms if term in messages[-1].content.lower()]
        
        logger.info("TV terms found: %s", tv_matches)
        logger.info("Issue terms found: %s", issue_matches)
        logger.info("Solution terms in final message: %s", solution_matches)
        
        # Check that the conversation mentions the core problem based on initial customer message
        # The initial message is: "I'm having problems with my Samsung TV."
        # RAG data also contains "flickering".
        all_content_lower = " ".join([msg.content.lower() for msg in messages])
        assert "samsung" in all_content_lower and "tv" in all_content_lower, \
            "Conversation should mention 'samsung' and 'tv' from initial query to show context was maintained."
        # Optionally, check for 'flicker' if it's expected to be brought up by RAG consistently
        # assert "flicker" in all_content_lower or "flickering" in all_content_lower, \
        #     "Conversation should ideally mention 'flicker'/'flickering' from RAG data."

        # Check the final agent message for basic structure
        final_agent_response = messages[-1].content
        logger.info(f"Final RAG agent response in multi-turn: {final_agent_response}")
        
        # Basic validation of the final agent response
        assert len(final_agent_response) > 20, "Final agent response should be more than 20 characters"
        assert any(phrase in final_agent_response.lower() for phrase in ["try", "you can"]), \
            "Final agent response should contain suggestions"

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
            # openai_api_key=openai_api_key, # create_vector_stores_from_conversations uses env var directly
        )
        assert customer_store is not None
        assert agent_store is not None

        # 2. Setup participants
        # The agent is our RagAgent
        agent = RagAgent(
            agent_vector_store=agent_store,
            # openai_api_key=openai_api_key, # RagAgent uses env var directly
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
        final_conversation_result = await runner.run()

        # 5. Assert results
        assert final_conversation_result is not None
        # Initial msg + customer msg + agent response
        assert len(final_conversation_result.conversation.messages) >= 3
        # Check if the agent's response (last message) contains relevant term
        # This is a simple check; could be made more robust like other tests if needed
        last_message_content = final_conversation_result.conversation.messages[-1].content.lower()
        assert "flicker" in last_message_content or "samsung" in last_message_content or "tv" in last_message_content, \
            f"Agent response should be relevant to TV flickering. Got: {final_conversation_result.conversation.messages[-1].content}"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rag_customer_and_agent_simulation_flow(
        self,
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
            # openai_api_key=openai_api_key, # create_vector_stores_from_conversations uses env var directly
        )
        assert customer_store is not None
        assert agent_store is not None

        # 2. Setup participants
        agent = RagAgent(
            agent_vector_store=agent_store,
            # openai_api_key=openai_api_key, # RagAgent uses env var directly
        )
        customer = RagCustomer(
            customer_vector_store=customer_store, # Using the actual customer_store from this test
            # openai_api_key=openai_api_key # RagCustomer uses env var directly
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
        final_conversation_result = await runner.run()

        # 5. Assert results
        assert final_conversation_result is not None
        assert len(final_conversation_result.conversation.messages) >= 3 # Initial, at least one customer, at least one agent

        # 6. Save conversation log (optional, good for debugging)
        log_dir = Path(__file__).parent.parent.parent / "data" / "simulation_logs"  # Go up to tests/ then data/simulation_logs
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "rag_customer_agent_simulation_log.json"

        conversation_log = []
        for msg in final_conversation_result.conversation.messages:
            conversation_log.append(
                {
                    "timestamp": msg.timestamp.isoformat(),
                    "sender": msg.sender.value,
                    "content": msg.content,
                }
            )

        with open(log_file_path, "w") as f:
            json.dump(conversation_log, f, indent=2)

        logger.info(f"Conversation log saved to: {log_file_path}")

        # Check progress handler
        assert len(progress_handler.messages_added) == len(final_conversation_result.conversation.messages)
        assert len(progress_handler.conversations_ended) == 1
