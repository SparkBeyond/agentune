#!/usr/bin/env python3
"""Utility functions for conversation simulator examples."""

import json
import logging

import pandas as pd
import nest_asyncio

from agentune.simulate.models import Conversation, Message, Outcome, ParticipantRole
from agentune.simulate.util.structure import converter


def load_conversations_from_csv(file_path: str) -> list[Conversation]:
    """Load conversations from a CSV file.
    
    Expected CSV format:
    - conversation_id: Unique identifier for each conversation
    - sender: Either "customer" or "agent"
    - content: The message content
    - timestamp: ISO format timestamp
    - outcome_name: Name of the conversation outcome
    - outcome_description: Description of the outcome
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of Conversation objects
    """
    df = pd.read_csv(file_path)
    
    # Group by conversation_id to reconstruct conversations
    conversations = []
    for conv_id, group in df.groupby('conversation_id'):
        # Sort by timestamp to ensure message order
        group = group.sort_values('timestamp')
        
        # Create messages
        messages = []
        for _, row in group.iterrows():
            # Convert sender string to ParticipantRole
            sender = ParticipantRole.CUSTOMER if row['sender'].lower() == 'customer' else ParticipantRole.AGENT
            
            message = Message(
                sender=sender,
                content=str(row['content']),
                timestamp=pd.to_datetime(row['timestamp']).to_pydatetime()
            )
            messages.append(message)
        
        # Create outcome from the first row (should be same for all messages in conversation)
        first_row = group.iloc[0]
        outcome = Outcome(
            name=str(first_row['outcome_name']),
            description=str(first_row['outcome_description'])
        )
        
        # Create conversation
        conversation = Conversation(
            messages=tuple(messages),
            outcome=outcome
        )
        conversations.append(conversation)
    
    return conversations


def extract_outcomes_from_conversations(conversations: list[Conversation]) -> list[Outcome]:
    """Extract unique outcomes from a list of conversations.
    
    Args:
        conversations: List of Conversation objects
        
    Returns:
        List of unique Outcome objects
    """
    seen_outcomes = set()
    unique_outcomes = []
    
    for conversation in conversations:
        if conversation.outcome and conversation.outcome.name not in seen_outcomes:
            seen_outcomes.add(conversation.outcome.name)
            unique_outcomes.append(conversation.outcome)
    
    return unique_outcomes


def load_conversations_from_json(file_path: str) -> list[Conversation]:
    """Load conversations from a JSON file (for compatibility with existing test data).
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of Conversation objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    for conv_data in data.get('conversations', []):
        conversation = converter.structure(conv_data, Conversation)
        conversations.append(conversation)
    
    return conversations


def conversations_to_csv(conversations: list[Conversation], output_path: str) -> None:
    """Convert conversations to CSV format for example purposes.
    
    Args:
        conversations: List of Conversation objects
        output_path: Path where to save the CSV file
    """
    rows = []
    
    for i, conversation in enumerate(conversations):
        conv_id = f"conv_{i+1:03d}"
        
        for message in conversation.messages:
            row = {
                'conversation_id': conv_id,
                'sender': message.sender.value,
                'content': message.content,
                'timestamp': message.timestamp.isoformat(),
                'outcome_name': conversation.outcome.name if conversation.outcome else 'unknown',
                'outcome_description': conversation.outcome.description if conversation.outcome else 'Unknown outcome'
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Converted {len(conversations)} conversations to {output_path}")


def setup_logging_and_asyncio() -> None:
    """Configure logging and asyncio for Jupyter notebooks."""
    # Configure basic logging to see simulation progress
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    # Fix asyncio event loop for Jupyter notebooks
    nest_asyncio.apply()
    
    print("✓ Logging configured")
    print("✓ Asyncio event loop configured for Jupyter")


def load_data_with_outcomes(csv_path: str) -> tuple[list[Conversation], tuple[Outcome, ...]]:
    """Load conversations from CSV and extract outcomes in one step.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Tuple of (conversations, outcomes) ready for simulation
    """
    print(f"Loading conversations from {csv_path}...")
    conversations = load_conversations_from_csv(csv_path)
    
    print(f"✓ Loaded {len(conversations)} conversations")
    print(f"✓ Sample conversation has {len(conversations[0].messages)} messages")
    
    # Extract outcomes
    unique_outcomes = extract_outcomes_from_conversations(conversations)
    outcomes_tuple = tuple(unique_outcomes)
    
    print(f"✓ Extracted {len(unique_outcomes)} unique outcomes")
    for outcome in unique_outcomes:
        print(f"  - {outcome.name}: {outcome.description}")
    
    return conversations, outcomes_tuple