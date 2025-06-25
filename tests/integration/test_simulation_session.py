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
from typing import override

import pytest
from cattrs.preconf.orjson import make_converter
from langchain_openai import ChatOpenAI

from conversation_simulator import SimulationSession
from conversation_simulator.intent_extraction.dummy import DummyIntentExtractor
from conversation_simulator.models import Conversation, Message, Outcome, Outcomes
from conversation_simulator.outcome_detection.base import OutcomeDetectionTest, OutcomeDetector
from conversation_simulator.participants.agent.config import AgentConfig
from conversation_simulator.participants.agent.zero_shot import ZeroShotAgentFactory
from conversation_simulator.participants.customer.zero_shot import ZeroShotCustomerFactory
from conversation_simulator.simulation.adversarial import ZeroShotAdversarialTester


# Configure cattrs for JSON serialization with datetime handling
converter = make_converter()


class SimpleOutcomeDetector(OutcomeDetector):
    """Mock outcome detector for testing."""

    @override
    async def detect_outcomes(
        self,
        instances: tuple[OutcomeDetectionTest, ...],
        possible_outcomes: Outcomes,
        return_exceptions: bool = True
    ) -> tuple[Outcome | None | Exception, ...]:
        """Mock outcome detection - return first available outcome."""
        if possible_outcomes.outcomes:
            return tuple(possible_outcomes.outcomes[0] for _ in instances)
        return tuple(None for _ in instances)

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
    def simulation_session(self, openai_model: ChatOpenAI, outcomes: Outcomes) -> SimulationSession:
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
            outcomes=outcomes,
            outcome_detector=outcome_detector,
            adversarial_tester=ZeroShotAdversarialTester(model=openai_model, max_concurrency=10),
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
        simulation_session: SimulationSession
    ):
        """Test the complete simulation pipeline end-to-end."""
        # Run the full pipeline with minimal data for testing
        session_result = await simulation_session.run_simulation(
            real_conversations=sample_conversations[:2],  # Use small subset
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
        
        # Load and convert data
        with open(dch2_dataset_path, encoding="utf-8") as f:
            data = json.load(f)
            conversations = converter.structure(data["conversations"], list[Conversation])
        
        # Verify we have the expected sample size
        assert len(conversations) == 100  # Sample size
        
        # Count outcomes for validation
        outcome_counts = {"resolved": 0, "unresolved": 0}
        for conv in conversations:
            if conv.outcome is not None:
                outcome_counts[conv.outcome.name] += 1
        
        # Verify resolution rate is reasonable (not 0% or 100%)
        total = sum(outcome_counts.values())
        resolution_rate = outcome_counts["resolved"] / total
        assert 0.1 < resolution_rate < 0.95  # Realistic range for customer service data
