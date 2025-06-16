"""Models package for conversation simulation."""

# Core domain models
from .conversation import Conversation
from .roles import ParticipantRole
from .intent import Intent
from .message import Message, MessageDraft
from .outcome import Outcome, Outcomes

# Simulation planning models
from .scenario import Scenario

# Result models  
from .results import ConversationResult, OriginalConversation, SimulatedConversation, SimulationSessionResult, SimulationAnalysisResult

# Analysis models
from .analysis import (
    OutcomeDistribution,
    OutcomeDistributionComparison, 
    MessageDistributionStats,
    MessageDistributionComparison,
    AdversarialEvaluationResult,
)

__all__ = [
    # Core domain models
    "Conversation",
    "Message", 
    "MessageDraft",
    "Intent",
    "Outcome",
    "Outcomes", 
    "ParticipantRole",
    # Simulation planning
    "Scenario",
    # Results
    "ConversationResult",
    "OriginalConversation", 
    "SimulatedConversation",
    "SimulationSessionResult",
    "SimulationAnalysisResult",
    # Analysis
    "OutcomeDistribution",
    "OutcomeDistributionComparison",
    "MessageDistributionStats", 
    "MessageDistributionComparison",
    "AdversarialEvaluationResult",
]
