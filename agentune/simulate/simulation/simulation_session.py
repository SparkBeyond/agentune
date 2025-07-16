"""Full simulation flow implementation."""

import asyncio
from datetime import datetime, timedelta
from collections.abc import Awaitable, Callable
import logging

from attrs import field, frozen


from agentune.simulate.simulation.progress import LoggingProgressCallback, ProgressCallback, ProgressCallbacks

from ..models.conversation import Conversation
from ..models.outcome import Outcomes
from ..models.scenario import Scenario
from ..models.results import (
    ConversationResult,
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

_logger = logging.getLogger(__name__)

@frozen
class SimulationSession:
    """Orchestrates the full simulation flow from real conversations to analysis.
    
    This class coordinates intent extraction, scenario generation, participant
    creation, conversation simulation, and result analysis.

    Attributes:
        outcomes: Legal outcome labels for this simulation run
        agent_factory: Factory for creating agent participants
        customer_factory: Factory for creating customer participants
        intent_extractor: Strategy for extracting intents from conversations
        outcome_detector: Strategy for detecting conversation outcomes
        adversarial_tester: Strategy for adversarial testing
        session_name: Human-readable name for this session
        session_description: Description of this simulation session
        max_messages: Maximum number of messages per conversation in simulation
        max_concurrent_conversations: Maximum number of conversations to run concurrently.
                                        Conversations will be processed in the order of the input list;
                                        however, we will simulate all conversations (with that many concurrent tasks)
                                        before analyzing all results (with, again, that many concurrent tasks).
        return_exceptions: If True, per-conversation exceptions will be logged and reported to the progress callback; 
                            the returned result will not include those conversations (not even in the 
                            total conversation count).
                            If False, any exception will be raised from this method and no information 
                            will be returned about other conversations.                              
        progress_callback: Callback for progress updates
        progress_log_interval: Interval at which to log progress. If no progress has been made,
                                nothing will be logged. (This is in addition to the custom progress callback.)
    """
    
    outcomes: Outcomes
    agent_factory: AgentFactory
    customer_factory: CustomerFactory
    intent_extractor: IntentExtractor
    outcome_detector: OutcomeDetector
    adversarial_tester: AdversarialTester
    session_name: str = "Simulation Session"
    session_description: str = "Automated conversation simulation"
    max_messages: int = 100
    max_concurrent_conversations: int = 10
    return_exceptions: bool = True
    progress_callback: ProgressCallback = field(factory=ProgressCallback)
    progress_log_interval: timedelta = timedelta(seconds=5)

    async def run_simulation(
        self,
        real_conversations: list[Conversation]
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
        _logger.info(f'Starting intent extraction on {len(original_conversations)} conversations')
        scenarios = await self._generate_scenarios(original_conversations)
        _logger.info(f'Finished extracting original intents; generated {len(scenarios)} scenarios')
        
        # Step 2: Run simulations for each scenario
        _logger.info(f'Starting conversation simulations ({self.max_concurrent_conversations=})')
        simulated_conversations_with_exceptions = await self._run_simulations(scenarios, self.max_concurrent_conversations)
        simulated_conversations = tuple(conv for conv in simulated_conversations_with_exceptions if isinstance(conv, SimulatedConversation))
        _logger.info(f'Finished simulating conversations; simulated {len(simulated_conversations)} conversations, '
                     f'with {len(simulated_conversations_with_exceptions) - len(simulated_conversations)} failures')
        
        # Step 3: Analyze results
        session_end = datetime.now()
        
        # Run comprehensive analysis
        _logger.info('Starting analysis of simulation results')
        analysis_result = await analyze_simulation_results(
            original_conversations=tuple(conv for conv in original_conversations),
            simulated_conversations=simulated_conversations,
            adversarial_tester=self.adversarial_tester,
            outcome_detector=self.outcome_detector,
            scenarios=scenarios,
            outcomes=self.outcomes,
            return_exceptions=self.return_exceptions,
        )
        _logger.info('Finished analyzing results')

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
        original_conversations: tuple[OriginalConversation, ...]
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

        intents = await self.intent_extractor.extract_intents(tuple(conv.conversation for conv in original_conversations),
                                                              return_exceptions=self.return_exceptions)

        for original_conv, intent in zip(original_conversations, intents):
            # Skip empty conversations
            if not original_conv.conversation.messages:
                continue

            # Use first message as initial message for scenario
            initial_message = MessageDraft(
                sender=original_conv.conversation.messages[0].sender,
                content=original_conv.conversation.messages[0].content,
            )

            if isinstance(intent, Exception):
                _logger.error(f"Error trying to extract intent for conversation {original_conv.id}", exc_info=intent)
                continue
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
    ) -> tuple[SimulatedConversation | Exception, ...]:
        """Run conversation simulations for all scenarios.
        
        Each simulated conversation gets a unique ID and maintains a mapping
        back to the scenario that generated it (and through that, to the
        original conversation).
        
        Args:
            scenarios: Scenarios to simulate
            
        Returns:
            Tuple of simulated conversations with proper ID mapping
        """

        logging_progress_callback = LoggingProgressCallback(self.progress_log_interval)
        all_callbacks = ProgressCallbacks((self.progress_callback, logging_progress_callback))
        
        def create_runner_run(scenario: Scenario) ->  Callable[[], Awaitable[ConversationResult]]:
            # Create participants - FullSimulationRunner will install intent as needed
            customer = self.customer_factory.create_participant()
            agent = self.agent_factory.create_participant()
            
            runner = FullSimulationRunner(
                customer=customer,
                agent=agent,
                initial_message=scenario.initial_message,
                intent=scenario.intent,
                outcomes=self.outcomes,
                outcome_detector=self.outcome_detector,
                max_messages=self.max_messages,
            )

            async def run() -> ConversationResult:
                all_callbacks.on_scenario_start(scenario)
                try:
                    result = await runner.run()
                    all_callbacks.on_scenario_complete(scenario, result)
                    return result
                except Exception as e:
                    all_callbacks.on_scenario_failed(scenario, e)
                    _logger.error(f'Error running simulation for scenario {scenario.id}', exc_info=e)
                    raise e
            return run
        
        # Create all runners
        runners = [create_runner_run(scenario) for scenario in scenarios]

        # Cancel the logging progress callback task when done or if abort early with an exception
        # (i.e. return_exceptions=False)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(logging_progress_callback.log_progress())

            all_callbacks.on_generated_scenarios(scenarios)
            
            # Run all simulations with bounded parallelism
            results = await tg.create_task(asyncutil.bounded_parallelism(
                runners, 
                max_concurrent_conversations,
                return_exceptions=self.return_exceptions
            ))

            all_callbacks.on_all_scenarios_complete()

        
        # Wrap results with SimulatedConversation
        simulated_conversations = tuple(
            result if isinstance(result, Exception) else
            SimulatedConversation(
                id=f"simulated_{i}",
                scenario_id=scenario.id,
                original_conversation_id=scenario.original_conversation_id,
                conversation=result.conversation,
            )
            for i, (scenario, result) in enumerate(zip(scenarios, results))
        )
        
        return simulated_conversations
    
