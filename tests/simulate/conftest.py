"""Configuration for pytest.

This file contains fixtures and configurations used by pytest.
"""
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from agentune.simulate.models import Conversation, Message, ParticipantRole


@pytest.fixture(scope="session", autouse=True)
def configure_logging() -> None:
    """Configure logging for all tests (unit and integration).
    
    This fixture runs automatically for all test sessions and configures
    logging to be useful for both debugging and viewing test output.
    """
    # Configure root logger with detailed format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True  # Override any existing configuration
    )
    
    # Reduce noise from third-party libraries during tests
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING) 
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # Add debug logging for specific modules if needed during development
    # logging.getLogger("agentune.simulate.participants").setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_chats_path(test_data_dir: Path) -> Path:
    """Return the path to the sample chats JSON file."""
    return test_data_dir / "sample_chats.json"


@pytest.fixture(scope="session")
def sample_chats_data(sample_chats_path: Path) -> list[dict[str, Any]]:
    """Load and return the sample chats data."""
    with open(sample_chats_path, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of conversations")
    return data


@pytest.fixture
def sample_conversation() -> Conversation:
    """Return a sample conversation for testing."""
    from datetime import datetime
    
    return Conversation(
        messages=(
            Message(
                content="Hello, I need help with my order",
                sender=ParticipantRole.CUSTOMER,
                timestamp=datetime(2024, 1, 1, 10, 0, 0)
            ),
            Message(
                content="I'd be happy to help. What's your order number?",
                sender=ParticipantRole.AGENT,
                timestamp=datetime(2024, 1, 1, 10, 0, 30)
            )
        )
    )
