"""Integration test configuration and fixtures."""

import os
import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from conversation_simulator.participants.agent.config import AgentConfig

# Load environment variables from .env file
load_dotenv()


def pytest_collection_modifyitems(config, items):
    """Automatically mark integration tests."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def openai_api_key():
    """OpenAI API key for integration tests."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping integration tests")
    return api_key


@pytest.fixture
def openai_model(openai_api_key):
    """Standard OpenAI model for tests."""
    return ChatOpenAI(
        api_key=SecretStr(openai_api_key),
        model="gpt-4o-mini",
        temperature=0.7
    )


@pytest.fixture
def support_agent_config():
    """Support agent configuration for tests."""
    return AgentConfig(
        company_name="TechSupport Pro",
        company_description="A technology support company helping customers with software and hardware issues",
        agent_role="Technical Support Representative"
    )


@pytest.fixture
def sales_agent_config():
    """Sales agent configuration for tests."""
    return AgentConfig(
        company_name="TechPro Solutions", 
        company_description="Premium business technology solutions provider specializing in enterprise software and cloud services",
        agent_role="Sales Representative"
    )
