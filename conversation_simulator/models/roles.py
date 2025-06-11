"""Participant roles for conversation simulation."""

from enum import Enum


class ParticipantRole(str, Enum):
    """Role of a conversation participant."""
    
    CUSTOMER = "customer"
    AGENT = "agent"
