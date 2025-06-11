"""Configuration for conversation agents."""

import attrs


@attrs.frozen
class AgentConfig:
    """Configuration for a conversation agent.
    
    This class holds the essential information needed to configure an agent
    for participating in conversations, including company context and role definition.
    """
    
    company_name: str
    """The name of the company the agent represents."""
    
    company_description: str
    """A detailed description of the company, its business, and relevant context."""
    
    agent_role: str
    """The specific role or position the agent plays (e.g., 'Sales Representative', 'Customer Support')."""