#!/usr/bin/env python3
"""
Common utilities for RAG2 experiments.

This module provides shared functions used by both build_vector_stores.py and run_simulation.py.
"""

import json
import logging
from pathlib import Path

from conversation_simulator.models import Conversation
from conversation_simulator.util.structure import converter

# Get module logger
logger = logging.getLogger(__name__)


def load_sample_conversations(data_file_path: Path) -> list[Conversation]:
    """Load sample conversations from a JSON file using cattrs.
    
    Args:
        data_file_path: Path to the conversation data JSON file
    
    Returns:
        List of sample Conversation objects
    """
    logger.info(f"Loading sample conversations from {data_file_path}")
    
    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    
    try:
        with data_file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats: 
        # 1. Direct list of conversations: [conv1, conv2, ...]
        # 2. Nested under 'conversations' key: {'conversations': [conv1, conv2, ...]}
        conversations_data = data.get('conversations', data)
        
        # Convert JSON data to Conversation objects using cattrs
        conversations: list[Conversation] = converter.structure(conversations_data, list[Conversation])
        logger.info(f"Loaded {len(conversations)} sample conversations")
        return conversations
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise ValueError(f"Failed to parse conversation data: {e}") from e
