"""Conversation fixtures for testing round-trip functionality."""

from __future__ import annotations
from datetime import datetime

import pytest

from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.message import MessageDraft
from conversation_simulator.models.roles import ParticipantRole
from ._conversation_util import from_drafts_to_conversation


@pytest.fixture
def conversation_customer_starts_with_passes(base_timestamp: datetime) -> Conversation:
    """A conversation where customer starts with passes creating consecutive agent messages.
    
    This simulates the result of a conversation where:
    1. Customer: "Hello, I need help" (initial)
    2. Agent turn: "How can I assist you?" 
    3. Customer turn: passes (None)
    4. Agent turn: "Is there anything specific you need help with?" (consecutive agent message)
    5. Customer turn: "Actually, never mind"
    6. Agent and Customer turns: both pass (conversation ends)
    """
    drafts = (
        MessageDraft(content="Hello, I need help", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="How can I assist you?", sender=ParticipantRole.AGENT),
        MessageDraft(content="Is there anything specific you need help with?", sender=ParticipantRole.AGENT),
        MessageDraft(content="Actually, never mind", sender=ParticipantRole.CUSTOMER),
    )
    return from_drafts_to_conversation(drafts)


@pytest.fixture  
def conversation_agent_starts_with_passes(base_timestamp: datetime) -> Conversation:
    """A conversation where agent starts with passes creating consecutive customer messages.
    
    This simulates the result of a conversation where:
    1. Agent: "Welcome! How can we help you today?" (initial)
    2. Customer turn: "I'm looking into an issue"
    3. Agent turn: passes (None) 
    4. Customer turn: "My order seems delayed" (consecutive customer message)
    5. Agent turn: "Let me check that for you"
    6. Customer and Agent turns: both pass (conversation ends)
    """
    drafts = (
        MessageDraft(content="Welcome! How can we help you today?", sender=ParticipantRole.AGENT),
        MessageDraft(content="I'm looking into an issue", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="My order seems delayed", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="Let me check that for you", sender=ParticipantRole.AGENT),
    )
    return from_drafts_to_conversation(drafts)
