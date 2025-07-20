# Coding Style Guide

## Python Version

- **Python 3.12+** required
- Use modern Python 3.12 features including enhanced typing and type parameter syntax
- No backward compatibility concerns

## Code Style

### PEP 8 with Adjustments
- Follow PEP 8 guidelines for naming conventions, indentation, and spacing
- **Line length**: 100 characters (relaxed from PEP 8's 79)
- Use ruff for automatic formatting and linting

### Type Hints
- Use modern Python 3.12 type hint syntax
- Avoid deprecated `typing` module aliases:
  - ✅ `list` not `typing.List`
  - ✅ `collections.abc.Iterable` not `typing.Iterable`
- Use new generic class syntax: `class Foo[T]:` instead of extending `Generic[T]`
- Prefer type bounds: `class Foo[F: Bar]:` over explicit `TypeVar` declarations

### Collection Types
- For function inputs: use `collections.abc.Iterable`
- For function outputs and dataclass attributes: use `tuple`
- Use `frozenset` instead of `set` for immutable collections
- Reason: tuples and frozensets are hashable and comparable

## Architecture Patterns

### Entity Definition
- **Required**: Use attrs library for all entities (dataclasses in public API)
- **Required**: Use `@attrs.frozen` for immutable entities
- Ensures hashability, thread safety, and comparison support

```python
import attrs

@attrs.frozen
class Message:
    content: str
    timestamp: datetime
    sender: ParticipantRole
```

### Abstract Interfaces
- Use `abc.ABC` and `@abstractmethod` for public interfaces
- Use `@override` decorator in implementations
- Separate public interfaces from private implementations

```python
from abc import ABC, abstractmethod
from typing import override

class Participant(ABC):
    @abstractmethod
    async def get_next_message(self, conversation: Conversation) -> Message | None:
        ...

class CustomerParticipant(Participant):
    @override
    async def get_next_message(self, conversation: Conversation) -> Message | None:
        # Implementation here
        ...
```

### Async Code
- **Required**: All external service calls (LLMs, HTTP, file I/O) must be async
- Use `asyncio`, not other async libraries
- Never block in async code - use `asyncio.to_thread` for blocking operations

## Documentation

### Docstrings
- **Required**: All public modules, classes, functions, and methods
- Follow PEP 257 conventions
- Include purpose, parameters, return values, and exceptions

```python
async def detect_outcome(self, conversation: Conversation) -> Outcome | None:
    """Detect if conversation has reached a specific outcome.
    
    Args:
        conversation: The conversation to analyze
        
    Returns:
        Detected outcome or None if no outcome reached
    """
```

## Code Quality Tools

Use ruff for linting and mypy for type checking. Run both before committing changes.

See [environment-setup.md](./environment-setup.md) for specific commands.

### Additional Standards
- Package management: Poetry
- Testing: pytest
- Logging: Python standard logging module

---

For environment setup instructions, see [environment-setup.md](./environment-setup.md).