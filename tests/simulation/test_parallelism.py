"""Tests for parallelism improvements in simulation session."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.intent import Intent
from conversation_simulator.models.message import Message, MessageDraft
from conversation_simulator.models.outcome import Outcome, Outcomes
from conversation_simulator.models.results import OriginalConversation, SimulatedConversation
from conversation_simulator.models.roles import ParticipantRole
from conversation_simulator.models.scenario import Scenario
from conversation_simulator.simulation.simulation_session import SimulationSession


class MockIntentExtractor:
    """Mock intent extractor that tracks call timing."""
    
    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.call_times = []
        self.call_count = 0
    
    async def extract_intent(self, conversation: Conversation) -> Intent | None:
        """Extract intent with simulated delay to test concurrency."""
        start_time = datetime.now()
        await asyncio.sleep(self.delay)
        end_time = datetime.now()
        
        self.call_times.append((start_time, end_time))
        self.call_count += 1
        
        return Intent(
            role=ParticipantRole.CUSTOMER,
            description=f"Test intent {self.call_count}"
        )


class MockFactory:
    """Mock factory for creating participants."""
    
    def __init__(self, role: ParticipantRole):
        self.role = role
    
    def create_participant(self):
        participant = MagicMock()
        participant.role = self.role
        return participant


class MockRunner:
    """Mock simulation runner that tracks timing."""
    
    call_times = []
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    async def run(self):
        """Simulate conversation execution with delay."""
        start_time = datetime.now()
        await asyncio.sleep(0.1)  # Simulate conversation time
        end_time = datetime.now()
        
        MockRunner.call_times.append((start_time, end_time))
        
        # Create a mock result
        result = MagicMock()
        result.conversation = Conversation(
            messages=tuple([
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="Test message",
                    timestamp=datetime.now()
                )
            ])
        )
        return result


class MockOutcomeDetector:
    """Mock outcome detector."""
    
    async def detect_outcome(self, conversation, intent, outcomes):
        return None


class MockAdversarialTester:
    """Mock adversarial tester."""
    
    async def identify_real_conversation(self, real_conv, sim_conv):
        return True


@pytest.fixture
def mock_outcomes():
    """Create mock outcomes for testing."""
    return Outcomes(
        outcomes=tuple([
            Outcome(name="resolved", description="Issue resolved"),
            Outcome(name="unresolved", description="Issue not resolved")
        ])
    )


@pytest.fixture
def sample_conversations():
    """Create sample conversations for testing."""
    return [
        Conversation(
            messages=tuple([
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content=f"Test message {i}",
                    timestamp=datetime.now()
                )
            ])
        )
        for i in range(3)
    ]


class TestParallelismImprovements:
    """Test the parallelism improvements in SimulationSession."""
    
    @pytest.mark.asyncio
    async def test_concurrent_intent_extraction(self, mock_outcomes, sample_conversations):
        """Test that intent extraction runs concurrently."""
        # Create intent extractor with delay
        intent_extractor = MockIntentExtractor(delay=0.1)
        
        # Create simulation session
        session = SimulationSession(
            outcomes=mock_outcomes,
            agent_factory=MockFactory(ParticipantRole.AGENT),
            customer_factory=MockFactory(ParticipantRole.CUSTOMER),
            intent_extractor=intent_extractor,
            outcome_detector=MockOutcomeDetector(),
            adversarial_tester=MockAdversarialTester(),
        )
        
        # Create original conversations
        original_conversations = tuple(
            OriginalConversation(id=f"original_{i}", conversation=conv)
            for i, conv in enumerate(sample_conversations)
        )
        
        # Measure time for scenario generation
        start_time = datetime.now()
        scenarios = await session._generate_scenarios(original_conversations)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds()
        
        # Verify all conversations were processed
        assert len(scenarios) == len(sample_conversations)
        assert intent_extractor.call_count == len(sample_conversations)
        
        # Verify concurrency: total time should be close to single delay (0.1s)
        # rather than sum of all delays (0.3s for 3 conversations)
        expected_sequential_time = len(sample_conversations) * 0.1
        assert total_time < expected_sequential_time * 0.8, \
            f"Expected concurrent execution (~0.1s), but took {total_time:.2f}s"
        
        # Verify calls happened concurrently by checking overlap in timing
        call_times = intent_extractor.call_times
        assert len(call_times) == 3
        
        # Check that calls started close to each other (concurrent execution)
        start_times = [start for start, _ in call_times]
        max_start_diff = max(start_times) - min(start_times)
        assert max_start_diff.total_seconds() < 0.05, \
            "Intent extraction calls should start almost simultaneously"
    
    @pytest.mark.asyncio
    async def test_concurrent_simulations(self, mock_outcomes, monkeypatch):
        """Test that simulations run concurrently."""
        # Reset call times
        MockRunner.call_times = []
        
        # Patch the FullSimulationRunner import
        from conversation_simulator.simulation import simulation_session
        monkeypatch.setattr(simulation_session, 'FullSimulationRunner', MockRunner)
        
        # Create simulation session
        session = SimulationSession(
            outcomes=mock_outcomes,
            agent_factory=MockFactory(ParticipantRole.AGENT),
            customer_factory=MockFactory(ParticipantRole.CUSTOMER),
            intent_extractor=MockIntentExtractor(),
            outcome_detector=MockOutcomeDetector(),
            adversarial_tester=MockAdversarialTester(),
        )
        
        # Create scenarios
        scenarios = tuple([
            Scenario(
                id=f"scenario_{i}",
                original_conversation_id=f"original_{i}",
                intent=Intent(role=ParticipantRole.CUSTOMER, description=f"Intent {i}"),
                initial_message=MessageDraft(
                    sender=ParticipantRole.CUSTOMER,
                    content=f"Test message {i}"
                )
            )
            for i in range(3)
        ])
        
        # Measure time for simulations
        start_time = datetime.now()
        simulated_conversations = await session._run_simulations(scenarios)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds()
        
        # Verify all scenarios were processed
        assert len(simulated_conversations) == len(scenarios)
        assert len(MockRunner.call_times) == len(scenarios)
        
        # Verify concurrency: total time should be close to single delay (0.1s)
        # rather than sum of all delays (0.3s for 3 simulations)
        expected_sequential_time = len(scenarios) * 0.1
        assert total_time < expected_sequential_time * 0.8, \
            f"Expected concurrent execution (~0.1s), but took {total_time:.2f}s"
        
        # Verify calls happened concurrently by checking overlap in timing
        call_times = MockRunner.call_times
        assert len(call_times) == 3
        
        # Check that calls started close to each other (concurrent execution)
        start_times = [start for start, _ in call_times]
        max_start_diff = max(start_times) - min(start_times)
        assert max_start_diff.total_seconds() < 0.05, \
            "Simulation calls should start almost simultaneously"
    
    @pytest.mark.asyncio
    async def test_intent_extraction_error_handling(self, mock_outcomes):
        """Test that intent extraction handles errors gracefully."""
        
        class FailingIntentExtractor:
            async def extract_intent(self, conversation):
                if "fail" in conversation.messages[0].content:
                    raise Exception("Intentional failure")
                return Intent(role=ParticipantRole.CUSTOMER, description="Success")
        
        # Create conversations, some of which will fail
        conversations = [
            Conversation(messages=tuple([
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="success message",
                    timestamp=datetime.now()
                )
            ])),
            Conversation(messages=tuple([
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="fail message",
                    timestamp=datetime.now()
                )
            ])),
            Conversation(messages=tuple([
                Message(
                    sender=ParticipantRole.CUSTOMER,
                    content="another success",
                    timestamp=datetime.now()
                )
            ]))
        ]
        
        # Create simulation session
        session = SimulationSession(
            outcomes=mock_outcomes,
            agent_factory=MockFactory(ParticipantRole.AGENT),
            customer_factory=MockFactory(ParticipantRole.CUSTOMER),
            intent_extractor=FailingIntentExtractor(),
            outcome_detector=MockOutcomeDetector(),
            adversarial_tester=MockAdversarialTester(),
        )
        
        # Create original conversations
        original_conversations = tuple(
            OriginalConversation(id=f"original_{i}", conversation=conv)
            for i, conv in enumerate(conversations)
        )
        
        # Run scenario generation - should not raise exception
        scenarios = await session._generate_scenarios(original_conversations)
        
        # Should only get scenarios for successful intent extractions
        assert len(scenarios) == 2  # Only 2 out of 3 should succeed
        
        # Verify the successful scenarios have proper intents
        for scenario in scenarios:
            assert scenario.intent is not None
            assert scenario.intent.description == "Success"