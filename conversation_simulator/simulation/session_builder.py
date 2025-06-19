"""Module providing builder pattern implementation for simulation sessions.

This module contains the SimulationSessionBuilder class which helps construct
SimulationSession instances with proper defaults and optional components.
"""

from typing import Self

import attrs
from langchain_core.language_models import BaseChatModel

from conversation_simulator.intent_extraction.base import IntentExtractor
from conversation_simulator.intent_extraction.zeroshot import ZeroshotIntentExtractor
from conversation_simulator.models.outcome import Outcomes
from conversation_simulator.outcome_detection.base import OutcomeDetector  
from conversation_simulator.outcome_detection.zeroshot import ZeroshotOutcomeDetector
from conversation_simulator.participants.agent.base import AgentFactory
from conversation_simulator.participants.customer.base import CustomerFactory
from conversation_simulator.simulation.adversarial.base import AdversarialTester
from conversation_simulator.simulation.adversarial.dummy import DummyAdversarialTester
from conversation_simulator.simulation.simulation_session import SimulationSession


@attrs.define
class SimulationSessionBuilder:
    """Builder for configuring and creating SimulationSession instances.
    
    This class provides a fluent interface for constructing SimulationSession objects
    with optional components and sensible defaults.
    
    Example:
        ```python
        session = (SimulationSessionBuilder(chat_model, agent_factory, customer_factory, outcomes)
                  .with_intent_extractor(custom_extractor)
                  .with_outcome_detector(custom_detector)
                  .build())
        ```
    """
    
    chat_model: BaseChatModel
    outcomes: Outcomes
    agent_factory: AgentFactory
    customer_factory: CustomerFactory
    session_name: str = "Simulation Session"
    session_description: str = "Automated conversation simulation"
    max_messages: int = 100
    _intent_extractor: IntentExtractor | None = None
    _outcome_detector: OutcomeDetector | None = None
    _adversarial_tester: AdversarialTester | None = None
    
    def with_intent_extractor(self, intent_extractor: IntentExtractor) -> Self:
        """Sets a custom intent extractor for the simulation session.
        
        Args:
            intent_extractor: The intent extractor implementation to use
            
        Returns:
            Self: The builder instance for method chaining
        """
        self._intent_extractor = intent_extractor
        return self
    
    def with_outcome_detector(self, outcome_detector: OutcomeDetector) -> Self:
        """Sets a custom outcome detector for the simulation session.
        
        Args:
            outcome_detector: The outcome detector implementation to use
            
        Returns:
            Self: The builder instance for method chaining
        """
        self._outcome_detector = outcome_detector
        return self
    
    def with_adversarial_tester(self, adversarial_tester: AdversarialTester) -> Self:
        """Sets a custom adversarial tester for the simulation session.
        
        Args:
            adversarial_tester: The adversarial tester implementation to use
            
        Returns:
            Self: The builder instance for method chaining
        """
        self._adversarial_tester = adversarial_tester
        return self
    
    def build(self) -> SimulationSession:
        """Creates a SimulationSession with the configured components.
        
        This method instantiates default components if they weren't explicitly set:
        - ZeroshotIntentExtractor for intent extraction
        - ZeroshotOutcomeDetector for outcome detection
        - DummyAdversarialTester for adversarial testing
        
        Returns:
            SimulationSession: A fully configured simulation session
        """
        return SimulationSession(
            outcomes=self.outcomes,
            agent_factory=self.agent_factory,
            customer_factory=self.customer_factory,
            intent_extractor=self._intent_extractor or ZeroshotIntentExtractor(self.chat_model),
            outcome_detector=self._outcome_detector or ZeroshotOutcomeDetector(self.chat_model),
            adversarial_tester=self._adversarial_tester or DummyAdversarialTester(),
            session_name=self.session_name,
            session_description=self.session_description,
            max_messages=self.max_messages,
        )
