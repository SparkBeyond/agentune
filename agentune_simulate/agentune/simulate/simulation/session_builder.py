"""Module providing builder pattern implementation for simulation sessions.

This module contains the SimulationSessionBuilder class which helps construct
SimulationSession instances with proper defaults and optional components.
"""

from typing import Self
from datetime import timedelta

import attrs
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from agentune.simulate.intent_extraction.zeroshot import ZeroshotIntentExtractor
from agentune.simulate.models.outcome import Outcomes
from agentune.simulate.outcome_detection.rag.rag import RAGOutcomeDetector
from agentune.simulate.participants.agent.real import RealAgentFactory
from agentune.simulate.participants.agent.rag.rag import RagAgentFactory
from agentune.simulate.participants.customer.rag.rag import RagCustomerFactory
from agentune.simulate.simulation.adversarial.zeroshot import ZeroShotAdversarialTester
from agentune.simulate.simulation.progress import ProgressCallback
from agentune.simulate.simulation.simulation_session import SimulationSession


@attrs.define
class SimulationSessionBuilder:
    """Builder for configuring and creating SimulationSession instances.
    
    This class provides a fluent interface for constructing SimulationSession objects
    with opinionated component choices and flexible model configurations:
    
    **Opinionated Components:**
    - Agent/Customer participants: Always uses RAG-based factories
    - Outcome detection: Always uses RAG-based detector
    - Intent extraction: Always uses zero-shot extractor
    - Adversarial testing: Always uses zero-shot tester
    
    **Flexible Models:**
    - Each component can use a different language model
    - Falls back to default_chat_model when specific models aren't specified

    Example:
        ```python
        session = (SimulationSessionBuilder(
                      default_chat_model=default_model,
                      outcomes=outcomes,
                      vector_store=vector_store
                  )
                  .with_customer_model(claude_model)
                  .with_outcome_detection_model(gemini_model)
                  .build())
        ```
    """
    
    default_chat_model: BaseChatModel
    outcomes: Outcomes
    vector_store: VectorStore

    # Optional model configurations - default to default_chat_model if not set
    agent_model: BaseChatModel | None = None
    customer_model: BaseChatModel | None = None
    outcome_detection_model: BaseChatModel | None = None
    intent_extraction_model: BaseChatModel | None = None
    adversarial_model: BaseChatModel | None = None

    # Optional factory overrides
    agent_factory: RealAgentFactory | None = None

    # Session configuration
    session_name: str = "Simulation Session"
    session_description: str = "Automated conversation simulation"
    max_messages: int = 100
    max_concurrent_conversations: int = 20
    return_exceptions: bool = True

    # Progress tracking
    progress_callback: ProgressCallback = attrs.field(factory=ProgressCallback)
    progress_log_interval: timedelta = timedelta(seconds=5)
    
    def with_agent_model(self, model: BaseChatModel) -> Self:
        """Sets the language model to use for agent participants.

        Args:
            model: The language model for agent responses

        Returns:
            Self: The builder instance for method chaining
        """
        self.agent_model = model
        return self

    def with_customer_model(self, model: BaseChatModel) -> Self:
        """Sets the language model to use for customer participants.

        Args:
            model: The language model for customer responses

        Returns:
            Self: The builder instance for method chaining
        """
        self.customer_model = model
        return self

    def with_outcome_detection_model(self, model: BaseChatModel) -> Self:
        """Sets the language model to use for outcome detection.

        Args:
            model: The language model for outcome detection

        Returns:
            Self: The builder instance for method chaining
        """
        self.outcome_detection_model = model
        return self
    
    def with_intent_extraction_model(self, model: BaseChatModel) -> Self:
        """Sets the language model to use for intent extraction.

        Args:
            model: The language model for intent extraction

        Returns:
            Self: The builder instance for method chaining
        """
        self.intent_extraction_model = model
        return self
    
    def with_adversarial_model(self, model: BaseChatModel) -> Self:
        """Sets the language model to use for adversarial testing.

        Args:
            model: The language model for adversarial testing

        Returns:
            Self: The builder instance for method chaining
        """
        self.adversarial_model = model
        return self

    def with_participant_models(self, model: BaseChatModel) -> Self:
        """Sets the same model for both agent and customer participants.

        Args:
            model: The language model for participant responses

        Returns:
            Self: The builder instance for method chaining
        """
        self.agent_model = model
        self.customer_model = model
        return self

    def with_agent_factory(self, factory: RealAgentFactory) -> Self:
        """Sets a custom agent factory to use instead of the default RAG agent.
        Note that this will override the agent_model setting if provided.

        Args:
            factory: The real agent factory for creating agent participants

        Returns:
            Self: The builder instance for method chaining
        """
        self.agent_factory = factory
        return self

    def build(self) -> SimulationSession:
        """Creates a SimulationSession with opinionated component choices.
        
        **Fixed Components (opinionated):**
        - RAG-based agent and customer factories
        - RAG-based outcome detector  
        - Zero-shot intent extractor and adversarial tester
        
        **Flexible Models:**
        - Each component uses its specified model or falls back to default_chat_model

        Returns:
            SimulationSession: A fully configured simulation session
        """
        max_concurrent_requests_in_batch = self.max_concurrent_conversations

        return SimulationSession(
            outcomes=self.outcomes,
            agent_factory=self.agent_factory or RagAgentFactory(
                model=self.agent_model or self.default_chat_model,
                agent_vector_store=self.vector_store
            ),
            customer_factory=RagCustomerFactory(
                model=self.customer_model or self.default_chat_model,
                customer_vector_store=self.vector_store
            ),
            intent_extractor=ZeroshotIntentExtractor(
                self.intent_extraction_model or self.default_chat_model,
                max_concurrency=max_concurrent_requests_in_batch
            ),
            outcome_detector=RAGOutcomeDetector(
                model=self.outcome_detection_model or self.default_chat_model,
                vector_store=self.vector_store
            ),
            adversarial_tester=ZeroShotAdversarialTester(
                self.adversarial_model or self.default_chat_model,
                max_concurrency=max_concurrent_requests_in_batch
            ),
            session_name=self.session_name,
            session_description=self.session_description,
            max_messages=self.max_messages,
            max_concurrent_conversations=self.max_concurrent_conversations,
            return_exceptions=self.return_exceptions,
            progress_callback=self.progress_callback,
            progress_log_interval=self.progress_log_interval,
        )
