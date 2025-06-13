"""Pytest configuration and fixtures for RAG integration tests."""

from datetime import datetime
import pytest

from conversation_simulator.models.intent import Intent
from conversation_simulator.models.outcome import Outcome, Outcomes


@pytest.fixture
def base_timestamp() -> datetime:
    """Return a fixed base timestamp for tests."""
    return datetime(2023, 5, 1, 10, 0, 0)


@pytest.fixture
def sample_intent() -> Intent:
    """Return a sample intent for testing."""
    from conversation_simulator.models.roles import ParticipantRole
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
