
"""Full simulation flow implementation."""

from __future__ import annotations
from datetime import datetime

from ..models.conversation import Conversation
from ..models.outcome import Outcomes
from ..models.scenario import Scenario
from ..models.results import (
    SimulationSessionResult,
    OriginalConversation,
    SimulatedConversation,
)
from ..models.message import MessageDraft
from ..intent_extraction.base import IntentExtractor
from ..participants.agent.base import AgentFactory
from ..participants.customer.base import CustomerFactory
from ..runners.full_simulation import FullSimulationRunner
from ..outcome_detection.base import OutcomeDetector
from .analysis import analyze_simulation_results


class SimulationSession:
    """Orchestrates the full simulation flow from real conversations to analysis.
    
    This class coordinates intent extraction, scenario generation, participant
    creation, conversation simulation, and result analysis.
    """
    
    def __init__(
        self,
        intent_extractor: IntentExtractor,
        agent_factory: AgentFactory,
        customer_factory: CustomerFactory,
        outcome_detector: OutcomeDetector,
        session_name: str = "Simulation Session",
        session_description: str = "Automated conversation simulation",
        max_messages: int = 100,
    ) -> None:
        """Initialize the simulation session.
        
        Args:
            intent_extractor: Strategy for extracting intents from conversations
            agent_factory: Factory for creating agent participants
            customer_factory: Factory for creating customer participants
            outcome_detector: Strategy for detecting conversation outcomes
            session_name: Human-readable name for this session
            session_description: Description of this simulation session
            max_messages: Maximum number of messages per conversation in simulation
        """
        self.intent_extractor = intent_extractor
        self.agent_factory = agent_factory
        self.customer_factory = customer_factory
        self.outcome_detector = outcome_detector
        self.session_name = session_name
        self.session_description = session_description
        self.max_messages = max_messages
    
    async def run_simulation(
        self,
        real_conversations: list[Conversation],
        outcomes: Outcomes,
    ) -> SimulationSessionResult:
        """Execute the full simulation flow.
        
        Args:
            real_conversations: Original conversations to base simulations on
            outcomes: Legal outcome labels for this simulation run
            
        Returns:
            Complete simulation results with analysis
        """
        session_start = datetime.now()
        
        # Step 0: Create original conversations with stable IDs early
        original_conversations = tuple(
            OriginalConversation(id=f"original_{i}", conversation=conv)
            for i, conv in enumerate(real_conversations)
        )
        
        # Step 1: Extract intents from conversations and generate scenarios
        scenarios = await self._generate_scenarios(original_conversations)
        
        # Step 2: Run simulations for each scenario
        simulated_conversations = await self._run_simulations(scenarios, outcomes)
        
        # Step 3: Analyze results
        session_end = datetime.now()
        session_duration = (session_end - session_start).total_seconds()
        
        # Run comprehensive analysis
        analysis_result = analyze_simulation_results(
            original_conversations=tuple(conv.conversation for conv in original_conversations),
            simulated_conversations=simulated_conversations
        )
        
        return SimulationSessionResult(
            session_name=self.session_name,
            session_description=self.session_description,
            started_at=session_start,
            completed_at=session_end,
            duration_seconds=session_duration,
            original_conversations=original_conversations,
            scenarios=scenarios,
            simulated_conversations=simulated_conversations,
            analysis_result=analysis_result,
        )
    
    async def _generate_scenarios(
        self,
        original_conversations: tuple[OriginalConversation, ...],
    ) -> tuple[Scenario, ...]:
        """Generate simulation scenarios from original conversations.
        
        Extract intents from conversations and create scenarios only for those
        where intent extraction succeeds. Each scenario gets a unique ID that
        references back to the original conversation ID.
        
        Args:
            original_conversations: Original conversations with stable IDs
            
        Returns:
            Tuple of scenarios for simulation
        """
        scenarios = []

        for original_conv in original_conversations:
            # Skip empty conversations
            if not original_conv.conversation.messages:
                continue

            # Use first message as initial message for scenario
            initial_message = MessageDraft(
                sender=original_conv.conversation.messages[0].sender,
                content=original_conv.conversation.messages[0].content,
            )

            # Extract intent from conversation
            intent = await self.intent_extractor.extract_intent(original_conv.conversation)
            
            if intent is None:
                continue  # Skip conversations where intent couldn't be extracted
            # Create scenario with simple unique ID
            scenario_id = f"scenario_{len(scenarios)}"
            
            scenario = Scenario(
                id=scenario_id,
                original_conversation_id=original_conv.id,
                intent=intent,
                initial_message=initial_message,
            )
            scenarios.append(scenario)
        
        return tuple(scenarios)
    
    async def _run_simulations(
        self,
        scenarios: tuple[Scenario, ...],
        outcomes: Outcomes,
    ) -> tuple[SimulatedConversation, ...]:
        """Run conversation simulations for all scenarios.
        
        Each simulated conversation gets a unique ID and maintains a mapping
        back to the scenario that generated it (and through that, to the
        original conversation).
        
        Args:
            scenarios: Scenarios to simulate
            outcomes: Legal outcome labels
            
        Returns:
            Tuple of simulated conversations with proper ID mapping
        """
        simulated_conversations = []
        
        for scenario in scenarios:
            # Create participants - FullSimulationRunner will install intent as needed
            customer = self.customer_factory.create_participant()
            agent = self.agent_factory.create_participant()
            
            # Create and run simulation
            runner = FullSimulationRunner(
                customer=customer,
                agent=agent,
                initial_message=scenario.initial_message,
                intent=scenario.intent,
                outcomes=outcomes,
                outcome_detector=self.outcome_detector,
                max_messages=self.max_messages,
            )
            
            result = await runner.run()
            
            # Create simulated conversation record with simple unique ID
            # The original_conversation_id field maintains the mapping back to source
            simulated_id = f"simulated_{len(simulated_conversations)}"
            simulated_conv = SimulatedConversation(
                id=simulated_id,          
                scenario_id=scenario.id,
                original_conversation_id=scenario.original_conversation_id,
                conversation=result.conversation,
            )
            simulated_conversations.append(simulated_conv)
        
        return tuple(simulated_conversations)


