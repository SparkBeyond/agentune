"""Scenario models for simulation planning."""

from __future__ import annotations
import attrs

from .intent import Intent
from .message import MessageDraft


@attrs.frozen
class Scenario:
    """A simulation scenario defining what conversation to simulate.
    
    Represents a planned simulation including the intent, how it should start,
    and which original conversation inspired it.
    
    ID Hierarchy:
    simulated_conversation_id -> scenario.id -> original_conversation_id
    This allows tracing any simulated conversation back to its source.
    """
    
    id: str  # Unique identifier for this scenario (NOT the same as simulated conversation ID)
    original_conversation_id: str  # Links back to the original conversation that inspired this scenario
    intent: Intent  # The extracted or assigned intent for this scenario
    initial_message: MessageDraft  # How the conversation should start
    
    def __str__(self) -> str:
        """String representation of the scenario."""
        return f"Scenario {self.id}: {self.intent} -> '{self.initial_message.content}'"
