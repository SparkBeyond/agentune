"""Shared fixtures for runner tests."""

from __future__ import annotations

import pytest

from conversation_simulator.models.conversation import Conversation
from conversation_simulator.models.message import MessageDraft
from conversation_simulator.models.roles import ParticipantRole
from ._conversation_util import from_drafts_to_conversation


@pytest.fixture
def conversation_customer_starts_with_passes() -> Conversation:
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
def conversation_agent_starts_with_passes() -> Conversation:
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


@pytest.fixture
def conversation_simple_back_and_forth() -> Conversation:
    """A simple conversation with perfect alternation (no passes).
    
    This simulates a straightforward conversation with no passes:
    1. Customer: "I need help with my account"
    2. Agent: "I can help you with that"
    3. Customer: "Great, what do you need?"
    4. Agent: "Can you provide your account number?"
    5. Customer: "It's 12345"
    6. Agent: "Perfect, I've updated your account"
    """
    drafts = (
        MessageDraft(content="I need help with my account", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="I can help you with that", sender=ParticipantRole.AGENT),
        MessageDraft(content="Great, what do you need?", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="Can you provide your account number?", sender=ParticipantRole.AGENT),
        MessageDraft(content="It's 12345", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="Perfect, I've updated your account", sender=ParticipantRole.AGENT),
    )
    return from_drafts_to_conversation(drafts)


@pytest.fixture
def conversation_multiple_consecutive_passes() -> Conversation:
    """A conversation with multiple consecutive messages from the same participant.
    
    This simulates a scenario where one participant passes multiple times:
    1. Customer: "Hello, I have an urgent issue"
    2. Agent: "I'm here to help"
    3. Agent: "Can you describe the issue?" (customer passed)
    4. Agent: "Please let me know what's wrong" (customer passed again)
    5. Customer: "Sorry, my connection was bad"
    6. Customer: "I'm having trouble with my order" (agent passed)
    """
    drafts = (
        MessageDraft(content="Hello, I have an urgent issue", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="I'm here to help", sender=ParticipantRole.AGENT),
        MessageDraft(content="Can you describe the issue?", sender=ParticipantRole.AGENT),
        MessageDraft(content="Please let me know what's wrong", sender=ParticipantRole.AGENT),
        MessageDraft(content="Sorry, my connection was bad", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="I'm having trouble with my order", sender=ParticipantRole.CUSTOMER),
    )
    return from_drafts_to_conversation(drafts)


@pytest.fixture
def conversation_agent_starts_simple() -> Conversation:
    """A simple conversation where the agent initiates.
    
    This simulates proactive agent engagement:
    1. Agent: "Hello! I see you're browsing our help section"
    2. Customer: "Yes, I'm looking for my order status"
    3. Agent: "I can help you find that information"
    4. Customer: "That would be great"
    """
    drafts = (
        MessageDraft(content="Hello! I see you're browsing our help section", sender=ParticipantRole.AGENT),
        MessageDraft(content="Yes, I'm looking for my order status", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="I can help you find that information", sender=ParticipantRole.AGENT),
        MessageDraft(content="That would be great", sender=ParticipantRole.CUSTOMER),
    )
    return from_drafts_to_conversation(drafts)


@pytest.fixture
def conversation_single_message() -> Conversation:
    """A conversation with just the initial message (immediate termination).
    
    This simulates a case where the conversation ends immediately:
    1. Customer: "Never mind, I figured it out"
    """
    drafts = (
        MessageDraft(content="Never mind, I figured it out", sender=ParticipantRole.CUSTOMER),
    )
    return from_drafts_to_conversation(drafts)


@pytest.fixture
def conversation_alternating_passes_pattern() -> Conversation:
    """A conversation with an alternating pattern of passes and messages.
    
    This simulates a complex interaction pattern:
    1. Customer: "I need help"
    2. Agent: "What can I help you with?" 
    3. Customer: "Actually, let me think..." (then customer passes next turn)
    4. Agent: "Take your time, I'm here when you're ready" (customer passed)
    5. Customer: "Okay, I'm ready now" (then agent passes next turn)
    6. Customer: "My question is about billing" (agent passed)
    7. Agent: "I can help with billing questions"
    """
    drafts = (
        MessageDraft(content="I need help", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="What can I help you with?", sender=ParticipantRole.AGENT),
        MessageDraft(content="Actually, let me think...", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="Take your time, I'm here when you're ready", sender=ParticipantRole.AGENT),
        MessageDraft(content="Okay, I'm ready now", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="My question is about billing", sender=ParticipantRole.CUSTOMER),
        MessageDraft(content="I can help with billing questions", sender=ParticipantRole.AGENT),
    )
    return from_drafts_to_conversation(drafts)
