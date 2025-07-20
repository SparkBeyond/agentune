"""Real agent participant base classes.

This module provides base classes for integrating real (external) agent systems
with the conversation simulation framework. Real agents are external systems that
implement their own logic and don't require intent descriptions to guide behavior.
"""

import abc
from typing import Self, override

from ..base import Agent, AgentFactory


class RealAgent(Agent):
    """Base class for real (external) agent implementations.
    
    Real agents are external systems that don't use intents for LLM guidance.
    They provide their own logic and can ignore intent descriptions.
    
    This class provides a sensible default implementation of the with_intent method
    that real agents inherit, eliminating the need for users to implement
    meaningless methods.
    
    Example:
        ```python
        class MyAPIAgent(RealAgent):
            async def get_next_message(self, conversation: Conversation) -> Optional[Message]:
                # Call your API here
                response = await my_agent_api.get_response(conversation)
                return Message(sender=self.role, content=response, timestamp=datetime.now())
        ```
    """
    @override
    def with_intent(self, intent_description: str) -> Self:
        """Real agents ignore intent descriptions and return themselves unchanged.
        
        This method is required by the Participant interface but real agents
        don't need to modify their behavior based on extracted intents since
        they implement their own logic.
        
        Args:
            intent_description: Natural language description of intent (ignored)
            
        Returns:
            Self (unchanged) for method chaining compatibility
        """
        return self


class RealAgentFactory(AgentFactory):
    """Abstract base factory for real (external) agent implementations.
    
    This factory follows the same pattern as other agent factories (RagAgentFactory,
    ZeroShotAgentFactory) to provide consistency in the API and potential extension
    points for shared real agent configuration.
    
    Example:
        ```python
        class MyAPIAgentFactory(RealAgentFactory):
            def __init__(self, api_endpoint: str, api_key: str):
                self.api_endpoint = api_endpoint
                self.api_key = api_key
                
            def create_participant(self) -> RealAgent:
                return MyAPIAgent(endpoint=self.api_endpoint, key=self.api_key)
        ```
    """
    @override
    @abc.abstractmethod
    def create_participant(self) -> RealAgent:
        """Create a real agent participant.
        
        Returns:
            Configured real agent instance
        """
        ...