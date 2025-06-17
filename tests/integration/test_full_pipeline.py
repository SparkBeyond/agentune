"""
Integration test for the full simulation pipeline using the converted DCH2 dataset.

This test validates the complete end-to-end workflow:
1. Load real conversation data
2. Extract intents from original conversations
3. Generate simulation scenarios
4. Create participant instances
5. Run simulations
6. Analyze results
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_openai import ChatOpenAI

from conversation_simulator import SimulationSession
from conversation_simulator.intent_extraction.dummy import DummyIntentExtractor
from conversation_simulator.models import Conversation, Intent, Message, Outcome, Outcomes
from conversation_simulator.outcome_detection.base import OutcomeDetector
from conversation_simulator.participants.agent.config import AgentConfig
from conversation_simulator.participants.agent.zero_shot import ZeroShotAgentFactory
from conversation_simulator.participants.customer.zero_shot import ZeroShotCustomerFactory
from conversation_simulator.simulation.adversarial import AdversarialTester


class SimpleOutcomeDetector(OutcomeDetector):
    """Mock outcome detector for testing."""
    
    async def detect_outcome(self, conversation: Conversation, intent: Intent, possible_outcomes: Outcomes) -> Outcome | None:
        """Mock outcome detection - return first available outcome."""
        if possible_outcomes.outcomes:
            return possible_outcomes.outcomes[0]
        return None

class DummyAdversarialTester(AdversarialTester):
    """Mock adversarial tester for testing purposes."""
    
    async def identify_real_conversation(
            self,
            conversation_1: Conversation,
            conversation_2: Conversation,
        ) -> int:
            # Simple heuristic: if conversation_1 has more messages, it's real
            return 0 if len(conversation_1.messages) > len(conversation_2.messages) else 1


@pytest.mark.integration
class TestFullPipelineIntegration:
    """Integration tests for the complete simulation pipeline."""

    @pytest.fixture
    def dch2_dataset_path(self) -> Path:
        """Path to the sampled DCH2 dataset."""
        return Path(__file__).parent.parent / "data" / "dch2_sampled_dataset.json"

    @pytest.fixture
    def sample_conversations(self, dch2_dataset_path: Path) -> list[Conversation]:
        """Load a sample of conversations from the DCH2 dataset."""
        with open(dch2_dataset_path, encoding="utf-8") as f:
            data = json.load(f)

        # Convert first 2 conversations to our models (very small for real LLM testing)
        conversations = []
        for conv_data in data["conversations"][:2]:
            messages = [
                Message(
                    sender=msg["sender"],
                    content=msg["content"],
                    timestamp=msg["timestamp"]
                )
                for msg in conv_data["messages"]
            ]
            
            outcome = Outcome(
                name=conv_data["outcome"]["name"],
                description=conv_data["outcome"]["description"]
            )
            
            conversation = Conversation(
                messages=tuple(messages),
                outcome=outcome
            )
            conversations.append(conversation)

        return conversations

    @pytest.fixture
    def outcomes(self) -> Outcomes:
        """Create test outcomes."""
        return Outcomes(
            outcomes=tuple([
                Outcome(name="resolved", description="Issue was successfully resolved"),
                Outcome(name="unresolved", description="Issue was not resolved")
            ])
        )

    @pytest.fixture
    def simulation_session(self, openai_model: ChatOpenAI) -> SimulationSession:
        """Create a simulation session with real LLM but simple outcome detection."""
        intent_extractor = DummyIntentExtractor()
        
        # Create agent config for factory
        agent_config = AgentConfig(
            company_name="Test Company",
            company_description="A test company for simulation",
            agent_role="Customer Service Agent"
        )
        
        agent_factory = ZeroShotAgentFactory(model=openai_model, agent_config=agent_config)
        customer_factory = ZeroShotCustomerFactory(model=openai_model)
        outcome_detector = SimpleOutcomeDetector()

        return SimulationSession(
            intent_extractor=intent_extractor,
            agent_factory=agent_factory,
            customer_factory=customer_factory,
            outcome_detector=outcome_detector,
            adversarial_tester=DummyAdversarialTester(),
        )

    @pytest.mark.asyncio
    async def test_simulation_session_creation(
        self, 
        simulation_session: SimulationSession
    ):
        """Test that simulation session is created properly."""
        assert simulation_session.intent_extractor is not None
        assert simulation_session.agent_factory is not None
        assert simulation_session.customer_factory is not None
        assert simulation_session.outcome_detector is not None

    @pytest.mark.asyncio
    async def test_full_pipeline_end_to_end(
        self,
        sample_conversations: list[Conversation],
        outcomes: Outcomes,
        simulation_session: SimulationSession
    ):
        """Test the complete simulation pipeline end-to-end."""
        # Run the full pipeline with minimal data for testing
        session_result = await simulation_session.run_simulation(
            real_conversations=sample_conversations[:2],  # Use small subset
            outcomes=outcomes
        )
        
        # Verify session result structure
        assert session_result is not None
        assert hasattr(session_result, 'original_conversations')
        assert hasattr(session_result, 'simulated_conversations')
        
        # Verify we have results
        assert len(session_result.original_conversations) == 2
        
        # Basic validation of structure
        for orig_conv in session_result.original_conversations:
            assert orig_conv.id.startswith("original_")
            assert orig_conv.conversation is not None

    def test_dataset_statistics(self, dch2_dataset_path: Path):
        """Test that the simplified dataset has expected properties."""

        with open(dch2_dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        
        # Verify simplified dataset structure
        assert "conversations" in data
        
        # Verify conversations have proper structure
        conversations = data["conversations"]
        assert len(conversations) == 100  # Sample size
        
        # Count outcomes for validation
        outcome_counts = {"resolved": 0, "unresolved": 0}
        for conv in conversations:
            outcome_counts[conv["outcome"]["name"]] += 1
        
        # Verify resolution rate is reasonable (not 0% or 100%)
        total = sum(outcome_counts.values())
        resolution_rate = outcome_counts["resolved"] / total
        assert 0.1 < resolution_rate < 0.95  # Realistic range for customer service data
        
        first_conv = conversations[0]
        assert "id" not in first_conv  # IDs removed in simplified format
        assert "messages" in first_conv
        assert "outcome" in first_conv
        assert len(first_conv["messages"]) > 0
        
        # Verify message structure
        first_message = first_conv["messages"][0]
        assert "sender" in first_message
        assert "content" in first_message
        assert "timestamp" in first_message
