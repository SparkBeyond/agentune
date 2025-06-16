"""Integration test configuration and fixtures."""

import os
import pytest
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from conversation_simulator.participants.agent.config import AgentConfig
from conversation_simulator.models.intent import Intent
from conversation_simulator.models.outcome import Outcome, Outcomes
from conversation_simulator.models.roles import ParticipantRole
from langchain_community.embeddings import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()


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


@pytest.fixture(scope="class")
def embedding_model(openai_api_key):
    """Standard OpenAI embedding model for tests."""
    
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=openai_api_key
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


@pytest.fixture
def base_timestamp() -> datetime:
    """Return a fixed base timestamp for tests."""
    return datetime(2023, 5, 1, 10, 0, 0)


@pytest.fixture
def sample_intent() -> Intent:
    """Return a sample intent for testing."""
    return Intent(
        role=ParticipantRole.CUSTOMER,
        description="Customer is having issues with their TV",
    )


@pytest.fixture
def sample_outcomes() -> Outcomes:
    """Return sample outcomes for testing."""
    return Outcomes(
        outcomes=tuple([
            Outcome(
                name="resolved",
                description="The TV issue was resolved",
            ),
            Outcome(
                name="escalated",
                description="The issue was escalated to technical support",
            ),
            Outcome(
                name="no_resolution",
                description="No resolution was reached",
            ),
        ])
    )
