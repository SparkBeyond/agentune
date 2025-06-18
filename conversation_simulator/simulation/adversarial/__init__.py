"""Adversarial testing for conversation simulation quality."""

from .base import AdversarialTester
from .dummy import DummyAdversarialTester

__all__ = [
    "AdversarialTester",
    "DummyAdversarialTester",
]
