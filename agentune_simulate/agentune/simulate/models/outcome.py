"""Outcome models for conversation simulation."""

import attrs


@attrs.frozen
class Outcome:
    """Conversation outcome with name and description."""
    
    name: str  # lowercase slug (e.g., "resolved", "escalated", "sale_completed")
    description: str  # human-readable description
    
    def __str__(self) -> str:
        """String representation of the outcome."""
        return f"{self.name}: {self.description}"


@attrs.frozen
class Outcomes:
    """Defines legal outcome labels for a simulation run."""
    
    outcomes: tuple[Outcome, ...]
    
    def __attrs_post_init__(self) -> None:
        """Validate uniqueness of outcome names."""
        names = [outcome.name for outcome in self.outcomes]
        if len(names) != len(set(names)):
            raise ValueError("Outcome names must be unique")
    
    @property
    def _outcome_dict(self) -> dict[str, Outcome]:
        """Create lookup dict (computed on each access - consider caching if needed)."""
        return {outcome.name: outcome for outcome in self.outcomes}
    
    def get_outcome_by_name(self, name: str) -> Outcome | None:
        """Get outcome by name, or None if not found."""
        return self._outcome_dict.get(name)
    
    @property
    def outcome_names(self) -> tuple[str, ...]:
        """Get all outcome names."""
        return tuple(outcome.name for outcome in self.outcomes)
