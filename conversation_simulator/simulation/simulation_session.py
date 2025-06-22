"""Full simulation flow implementation."""

from __future__ import annotations
import asyncio
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
    ) -> SimulationSessionResult:
        """Execute the full simulation flow.
        
        Args:
            real_conversations: Original conversations to base simulations on
         
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
        simulated_conversations = await self._run_simulations(scenarios)
         
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
        # Filter out empty conversations first
        valid_conversations = [
            conv for conv in original_conversations
            if conv.conversation.messages
        ]
        
        if not valid_conversations:
            return tuple()
        
        # Try to use batch processing if available, otherwise fall back to sequential
        intents = await self._extract_intents_batch(valid_conversations)
        
        scenarios = []
        for conv, intent in zip(valid_conversations, intents):
            if intent is None:
                continue  # Skip conversations where intent couldn't be extracted
            
            # Use first message as initial message for scenario
            initial_message = MessageDraft(
                sender=conv.conversation.messages[0].sender,
                content=conv.conversation.messages[0].content,
            )

            # Create scenario with simple unique ID
            scenario_id = f"scenario_{len(scenarios)}"
            
            scenario = Scenario(
                id=scenario_id,
                original_conversation_id=conv.id,
                intent=intent,
                initial_message=initial_message,
            )
            scenarios.append(scenario)
        
        return tuple(scenarios)
    
    async def _extract_intents_batch(
        self,
        conversations: list[OriginalConversation],
    ) -> list[Intent | None]:
        """Extract intents from multiple conversations, using batch processing if available.
        
        Args:
            conversations: List of conversations to extract intents from
            
        Returns:
            List of extracted intents (or None for failed extractions)
        """
        # Check if the intent extractor supports batch processing
        if hasattr(self.intent_extractor, '_chain') and hasattr(self.intent_extractor._chain, 'abatch'):
            # Use LangChain's abatch for concurrent processing
            try:
                from .prompts import format_conversation
                
                # Prepare batch inputs
                batch_inputs = []
                for conv in conversations:
                    formatted_conversation = format_conversation(conv.conversation)
                    batch_inputs.append({"conversation": formatted_conversation})
                
                # Process batch
                results = await self.intent_extractor._chain.abatch(batch_inputs)
                
                # Convert results to intents
                intents = []
                for result in results:
                    try:
                        intent = result.to_intent() if hasattr(result, 'to_intent') else None
                        intents.append(intent)
                    except Exception:
                        intents.append(None)
                
                return intents
                
            except Exception:
                # Fall back to sequential processing if batch fails
                pass
        
        # Fall back to sequential processing using asyncio.gather for concurrency
        tasks = [
            self.intent_extractor.extract_intent(conv.conversation)
            for conv in conversations
        ]
        
        # Use gather with return_exceptions=True to handle individual failures
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        intents = []
        for result in results:
            if isinstance(result, Exception):
                intents.append(None)
            else:
                intents.append(result)
        
        return intents
    
    async def _run_simulations(
        self,
        scenarios: tuple[Scenario, ...],
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
        
        async def run_single_simulation(scenario: Scenario, sim_index: int) -> SimulatedConversation:
            """Run a single simulation and return the result."""
            # Create participants - FullSimulationRunner will install intent as needed
            customer = self.customer_factory.create_participant()
            agent = self.agent_factory.create_participant()
            
            # Create and run simulation
            runner = FullSimulationRunner(
                customer=customer,
                agent=agent,
                initial_message=scenario.initial_message,
                intent=scenario.intent,
                outcomes=self.outcomes,
                outcome_detector=self.outcome_detector,
                max_messages=self.max_messages,
            )
            
            result = await runner.run()
            
            # Create simulated conversation record with simple unique ID
            # The original_conversation_id field maintains the mapping back to source
            simulated_id = f"simulated_{sim_index}"
            return SimulatedConversation(
                id=simulated_id,          
                scenario_id=scenario.id,
                original_conversation_id=scenario.original_conversation_id,
                conversation=result.conversation,
            )
        
        # Run all simulations concurrently using asyncio.gather
        simulation_tasks = [
            run_single_simulation(scenario, i) 
            for i, scenario in enumerate(scenarios)
        ]
        
        simulated_conversations = await asyncio.gather(*simulation_tasks)
        
        return tuple(simulated_conversations)


