"""Test data for indexing and retrieval tests.

This module contains mock conversations and other test data
that can be reused across multiple test files.
"""

from datetime import datetime
from conversation_simulator.models import Conversation, Message, ParticipantRole, Outcome


def create_test_conversations():
    """Create test conversations with varied patterns for testing."""
    # Create a set of conversations for testing
    conversations = []
    
    # Create outcomes for testing
    resolved_outcome = Outcome(name="resolved", description="Issue was resolved")
    
    # 1. Simple single-turn conversation (complete)
    simple_complete_conversation = Conversation(
        messages=(
            Message(sender=ParticipantRole.CUSTOMER, content="How do I reset my password?", timestamp=datetime(2023, 1, 1, 10, 0)),
            Message(sender=ParticipantRole.AGENT, content="You can reset your password by clicking the 'Forgot Password' link on the login page.", timestamp=datetime(2023, 1, 1, 10, 1)),
        ),
        outcome=resolved_outcome
    )
    conversations.append(simple_complete_conversation)
    
    # 2. Simple single-turn conversation (incomplete)
    simple_incomplete_conversation = Conversation(
        messages=(
            Message(sender=ParticipantRole.CUSTOMER, content="Where can I find your pricing information?", timestamp=datetime(2023, 1, 1, 11, 0)),
        ),
        outcome=None
    )
    conversations.append(simple_incomplete_conversation)
    
    # 3. Multi-turn conversation (complete)
    multi_turn_complete_conversation = Conversation(
        messages=(
            Message(sender=ParticipantRole.CUSTOMER, content="I'm having trouble with my account.", timestamp=datetime(2023, 1, 2, 10, 0)),
            Message(sender=ParticipantRole.AGENT, content="I'm sorry to hear that. What specific issue are you experiencing?", timestamp=datetime(2023, 1, 2, 10, 1)),
            Message(sender=ParticipantRole.CUSTOMER, content="I can't log in, it says my password is incorrect but I'm sure it's right.", timestamp=datetime(2023, 1, 2, 10, 2)),
            Message(sender=ParticipantRole.AGENT, content="Let me help you reset your password. I've sent a reset link to your email.", timestamp=datetime(2023, 1, 2, 10, 3)),
        ),
        outcome=resolved_outcome
    )
    conversations.append(multi_turn_complete_conversation)
    
    # 4. Conversation expecting an agent response
    conversation_expecting_agent = Conversation(
        messages=(
            Message(sender=ParticipantRole.CUSTOMER, content="Can you tell me about your return policy?", timestamp=datetime(2023, 1, 3, 14, 0)),
            Message(sender=ParticipantRole.AGENT, content="Let me check our inventory.", timestamp=datetime(2023, 1, 4, 14, 1)),
            Message(sender=ParticipantRole.CUSTOMER, content="Thanks, I'll wait for your response.", timestamp=datetime(2023, 1, 4, 14, 2)),
        ),
        outcome=None
    )
    conversations.append(conversation_expecting_agent)
    
    # 5. Conversation expecting a customer response
    conversation_expecting_customer = Conversation(
        messages=(
            Message(sender=ParticipantRole.CUSTOMER, content="What payment methods do you accept?", timestamp=datetime(2023, 1, 5, 15, 0)),
            Message(sender=ParticipantRole.AGENT, content="We accept credit cards, PayPal, and bank transfers. Which would you prefer to use?", timestamp=datetime(2023, 1, 5, 15, 1)),
        ),
        outcome=None
    )
    conversations.append(conversation_expecting_customer)
    
    # 6-10. Additional conversations to ensure we have enough AGENT examples
    for i in range(6, 11):
        agent_example = Conversation(
            messages=(
                Message(sender=ParticipantRole.CUSTOMER, content=f"I need help with issue {i}.", timestamp=datetime(2023, 1, i, 16, 0)),
                Message(sender=ParticipantRole.AGENT, content=f"I'd be happy to help you with issue {i}. Let me look into that for you.", timestamp=datetime(2023, 1, i, 16, 1)),
            ),
            outcome=resolved_outcome
        )
        conversations.append(agent_example)
    
    # 11-15. Additional conversations to ensure we have enough CUSTOMER examples  
    for i in range(11, 16):
        customer_example = Conversation(
            messages=(
                Message(sender=ParticipantRole.AGENT, content=f"How can I assist you today with matter {i}?", timestamp=datetime(2023, 1, i, 17, 0)),
                Message(sender=ParticipantRole.CUSTOMER, content=f"I'm experiencing problem {i} and need guidance.", timestamp=datetime(2023, 1, i, 17, 1)),
            ),
            outcome=resolved_outcome
        )
        conversations.append(customer_example)
    
    return conversations
