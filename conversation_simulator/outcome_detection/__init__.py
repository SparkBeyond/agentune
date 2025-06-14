"""Outcome detection module for conversation analysis."""

from .base import OutcomeDetector
from .zeroshot import ZeroshotOutcomeDetector

__all__ = ["OutcomeDetector", "ZeroshotOutcomeDetector"]
