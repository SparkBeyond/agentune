"""Agentune Simulate - A library for simulating customer-agent conversations."""

# Core models
from .models.conversation import Conversation
from .models.roles import ParticipantRole
from .models.intent import Intent
from .models.message import Message, MessageDraft
from .models.outcome import Outcome, Outcomes

# Result models
from .models.results import ConversationResult, SimulationSessionResult, OriginalConversation, SimulatedConversation, SimulationAnalysisResult

# Analysis models  
from .models.analysis import OutcomeDistribution, MessageDistributionStats, AdversarialEvaluationResult

# Simulation planning
from .models.scenario import Scenario

# Base classes
from .participants.base import Participant, ParticipantFactory
from .runners.base import Runner

# Runners
from .runners.full_simulation import FullSimulationRunner

# Simulation orchestration
from .simulation import SimulationSession
from .simulation.session_builder import SimulationSessionBuilder

__version__ = "0.1.0"

__all__ = [
    # Core models
    "Conversation",
    "Message", 
    "MessageDraft",
    "Intent",
    "Outcome",
    "Outcomes", 
    "ParticipantRole",
    # Result models
    "ConversationResult",
    "SimulationSessionResult",
    "OriginalConversation",
    "SimulatedConversation",
    "SimulationAnalysisResult", 
    # Analysis models
    "OutcomeDistribution",
    "MessageDistributionStats", 
    "AdversarialEvaluationResult",
    # Simulation planning
    "Scenario",
    # Base classes
    "Participant",
    "ParticipantFactory",
    "Runner",
    # Runners
    "FullSimulationRunner",
    # Simulation orchestration
    "SimulationSession",
    "SimulationSessionBuilder",
]
