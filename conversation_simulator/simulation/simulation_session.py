"""Full simulation flow implementation."""

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
from .adversarial import AdversarialTester
from ..util import asyncutil

class SimulationSession:
    """Orchestrates the full simulation flow from real conversations to analysis.
    
    This class coordinates intent extraction, scenario generation, participant
    creation, conversation simulation, and result analysis.
    """
    
    def __init__(
        self,
        outcomes: Outcomes,
        agent_factory: AgentFactory,
        customer_factory: CustomerFactory,
        intent_extractor: IntentExtractor,
        outcome_detector: OutcomeDetector,
        adversarial_tester: AdversarialTester,
        session_name: str = "Simulation Session",
        session_description: str = "Automated conversation simulation",
        max_messages: int = 100,
    ) -> None:
        """Initialize the simulation session.
        
        Args:
            outcomes: Legal outcome labels for this simulation run
            agent_factory: Factory for creating agent participants
            customer_factory: Factory for creating customer participants
            intent_extractor: Strategy for extracting intents from conversations
            outcome_detector: Strategy for detecting conversation outcomes
            adversarial_tester: Strategy for adversarial testing
            session_name: Human-readable name for this session
            session_description: Description of this simulation session
            max_messages: Maximum number of messages per conversation in simulation
        """
        self.outcomes = outcomes
        self.agent_factory = agent_factory
        self.customer_factory = customer_factory
        self.intent_extractor = intent_extractor
        self.outcome_detector = outcome_detector
        self.adversarial_tester = adversarial_tester
        self.session_name = session_name
        self.session_description = session_description
        self.max_messages = max_messages
    
    async def run_simulation(
        self,
        real_conversations: list[Conversation],
        max_concurrent_conversations: int = 10
    ) -> SimulationSessionResult:
        """Execute the full simulation flow.
        
        Args:
            real_conversations: Original conversations to base simulations on
            max_concurrent_conversations: Maximum number of conversations to run concurrently.
                                          Conversations will be processed in the order of the input list.
         
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
        simulated_conversations = await self._run_simulations(scenarios, max_concurrent_conversations)
         
        # Step 3: Analyze results
        session_end = datetime.now()
        
        # Run comprehensive analysis
        analysis_result = await analyze_simulation_results(
            original_conversations=tuple(conv.conversation for conv in original_conversations),
            simulated_conversations=simulated_conversations,
            adversarial_tester=self.adversarial_tester,
        )
        
        return SimulationSessionResult(
            session_name=self.session_name,
            session_description=self.session_description,
            started_at=session_start,
            completed_at=session_end,
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
        max_concurrent_conversations: int
    ) -> tuple[SimulatedConversation, ...]:
        """Run conversation simulations for all scenarios.
        
        Each simulated conversation gets a unique ID and maintains a mapping
        back to the scenario that generated it (and through that, to the
        original conversation).
        
        Args:
            scenarios: Scenarios to simulate
            
        Returns:
            Tuple of simulated conversations with proper ID mapping
        """
        
        def create_runner(scenario: Scenario) -> FullSimulationRunner:
            # Create participants - FullSimulationRunner will install intent as needed
            customer = self.customer_factory.create_participant()
            agent = self.agent_factory.create_participant()
            
            return FullSimulationRunner(
                customer=customer,
                agent=agent,
                initial_message=scenario.initial_message,
                intent=scenario.intent,
                outcomes=self.outcomes,
                outcome_detector=self.outcome_detector,
                max_messages=self.max_messages,
            )
        
        # Create all runners
        runners = [create_runner(scenario) for scenario in scenarios]
        
        # Run all simulations with bounded parallelism
        results = await asyncutil.bounded_parallelism(
            [runner.run for runner in runners], 
            max_concurrent_conversations,
            return_exceptions=False
        )
        
        # Wrap results with SimulatedConversation
        simulated_conversations = tuple(
            SimulatedConversation(
                id=f"simulated_{i}",
                scenario_id=scenario.id,
                original_conversation_id=scenario.original_conversation_id,
                conversation=result.conversation,
            )
            for i, (scenario, result) in enumerate(zip(scenarios, results))
        )
        
        return simulated_conversations
    
