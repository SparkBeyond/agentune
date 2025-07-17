# Agentune Simulate Design

## Overview

The Agentune Simulator supports two primary flows:
1. **Full Simulation**: Both customer and agent are simulated
2. **Hybrid Simulation**: Simulated customer with a real agent

## Core Components

### 1. Participant Interface

The base interface for all simulated conversation participants (simulated agents and customers).

```python
class Participant(abc.ABC):
    @abstractmethod
    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """
        Generate the next message based on conversation history.
        
        Args:
            conversation: The conversation history up to this point
            
        Returns:
            Message with timestamp, or None if participant is finished
        """
        ...
```

### 2. Message Structure

```python
@attrs.frozen
class MessageDraft:
    """Message content without timestamp - to be assigned during simulation."""
    content: str
    sender: ParticipantRole  # AGENT or CUSTOMER

@attrs.frozen
class Message: 
    content: str
    timestamp: datetime
    sender: ParticipantRole  # AGENT or CUSTOMER
```

### 3. Channel Interface

Used for communication with real agents.

```python
class Channel(abc.ABC):
    @abstractmethod
    async def create_session(self) -> Session:
        """Create a new conversation session."""
        ...

class Session(abc.ABC):
    session_id: str
    
    @abstractmethod
    async def get_conversation(self) -> Conversation:
        """Get the conversation history so far."""
        ...
    
    @abstractmethod
    async def send(self, message: Message) -> None:
        """Send a message to the real agent."""
        ...
    
    @abstractmethod
    async def subscribe(self) -> AsyncIterator[Message]:
        """Subscribe to incoming messages from the agent."""
        ...
```

### 4. Intent Structure

Represents a participant's goal or purpose for the conversation.

```python
@attrs.frozen
class Intent:
    """A participant's goal/purpose for the conversation."""
    role: ParticipantRole  # Who has this intent (CUSTOMER or AGENT)
    description: str  # 1-2 sentences describing what they want to achieve
    
    # Examples:
    # Intent(role=ParticipantRole.CUSTOMER, description="Cancel subscription due to billing issues")
    # Intent(role=ParticipantRole.CUSTOMER, description="Set up two-factor authentication on account")
    # Intent(role=ParticipantRole.AGENT, description="Upsell customer to premium plan during support interaction")
```

### 5. Outcome and Outcome Schema

Represents conversation outcomes for analysis and evaluation.

```python
@attrs.frozen
class Outcome:
    """Conversation outcome with name and description."""
    name: str  # lowercase slug (e.g., "resolved", "escalated", "sale_completed")
    description: str  # human-readable description
    
@attrs.frozen  
class Outcomes:
    """Defines legal outcome labels for a simulation run."""
    outcomes: tuple[Outcome, ...]  # list of valid outcomes
    
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
    
    # Example:
    # Outcomes(outcomes=(
    #     Outcome(name="resolved", description="Customer issue was fully resolved"),
    #     Outcome(name="escalated", description="Issue was escalated to supervisor"),
    #     Outcome(name="sale_completed", description="Customer completed a purchase")
    # ))
```

### 6. Conversation Structure

```python
@attrs.frozen
class Conversation:
    """Complete conversation with messages and optional outcome."""
    messages: tuple[Message, ...]
    outcome: Outcome | None = None
```

### 7. Simulation Result

Result object returned by conversation runners.

```python
@attrs.frozen
class ConversationResult:
    """Result of simulating a single conversation."""
    conversation: Conversation
    duration: timedelta = timedelta(seconds=0)
```

### 8. Outcome Detection

Outcome detection is handled by pluggable detector strategies that analyze conversations to determine if specific outcomes have been reached.

```python
class OutcomeDetector(abc.ABC):
    """Abstract base class for outcome detection strategies."""
    
    @abc.abstractmethod
    async def detect_outcome(
        self, 
        conversation: Conversation, 
        intent: Intent, 
        possible_outcomes: Outcomes
    ) -> Outcome | None:
        ...
```
The runner receives an `OutcomeDetector` instance and calls it during conversation execution to check if conversation has reached any defined outcome.

## Flow Implementations

### 1. Full Simulation Flow

Orchestrated by `FullSimulationRunner` - handles conversations between two simulated participants.


```python
class FullSimulationRunner(Runner):
    """Runs conversations with both simulated customer and agent.
    
    Single-use runner that manages conversation state internally.
    Provides progress tracking capabilities for conversations.
    """
    
    customer: Participant
    agent: Participant
    initial_message: MessageDraft
    intent: Intent
    outcomes: Outcomes
    outcome_detector: OutcomeDetector  # Strategy for detecting conversation completion
    max_messages: int = 100
    max_messages_after_outcome: int = 5  # Allow goodbye messages after outcome
    base_timestamp: datetime | None = None  # If None, use current time when run() starts
    progress_handler: ProgressHandler | None = None  # Event callbacks for progress tracking
    
    async def run(self) -> ConversationResult:
        """Execute the full simulation conversation.
        
        Returns:
            ConversationResult with conversation history and metadata
        """
        # Implementation: turn-based sequential message generation
        # - Participants strictly alternate turns
        # - Each participant in turn can respond with a message or pass (None)
        # - Conversation ends when both participants pass consecutively
        # - Uses outcome_detector to check for conversation completion after each message
        # - Supports configurable post-outcome message handling
    
    @property
    def conversation(self) -> Conversation:
        """Get current conversation state (read-only access)."""
        ...
    
    @property
    def is_complete(self) -> bool:
        """Check if the simulation has completed."""
        return self._is_complete
    
    def get_progress(self) -> dict[str, Any]:
        """Get current progress information."""
        return {
            "message_count": len(self._conversation.messages),
            "max_messages": self.max_messages,
            "is_complete": self._is_complete,
            "outcome": self._conversation.outcome.name if self._conversation.outcome else None,
            "elapsed_seconds": (
                (datetime.now() - self._start_time).total_seconds() 
                if self._start_time else 0
            ),
        }
```

**Key characteristics:**
- Turn-based sequential message generation - simulation runner strictly alternate between customer and agent
- Each participant can either respond with a message or pass (return None)
- Conversation ends when both participants pass consecutively
- Deterministic message ordering based on strict turn order
- No real-time constraints - can run at maximum speed
- Both participants are AI-driven

**Usage Example:**
```python
# Define outcome schema
outcomes = Outcomes(outcomes=(
    Outcome(name="resolved", description="Customer issue was fully resolved"),
    Outcome(name="escalated", description="Issue was escalated to supervisor"),
    Outcome(name="abandoned", description="Customer left without resolution")
))

# Create participants
customer = CustomerParticipant(...)
agent = AgentParticipant(...)

# Create and run simulation
runner = FullSimulationRunner(
    customer=customer, 
    agent=agent, 
    initial_message=MessageDraft(content="Hello, I need help", sender=ParticipantRole.CUSTOMER),
    intent=support_intent,
    outcomes=outcomes,
    max_messages=50
)
result = await runner.run()

print(f"Conversation completed in {result.duration.total_seconds():.2f}s")
print(f"Messages exchanged: {len(result.conversation.messages)}")
if result.conversation.outcome:
    print(f"Outcome: {result.conversation.outcome.name} - {result.conversation.outcome.description}")

# Progress tracking example
def progress_handler(conversation: Conversation, progress: dict[str, Any]) -> None:
    print(f"Progress: {progress['message_count']}/{progress['max_messages']} messages")
    if progress['outcome']:
        print(f"Outcome reached: {progress['outcome']}")

runner_with_progress = FullSimulationRunner(
    customer=customer,
    agent=agent,
    initial_message=MessageDraft(content="Hello, I need help", sender=ParticipantRole.CUSTOMER),
    intent=support_intent,
    outcomes=outcomes,
    progress_callback=progress_handler
)
```

## Design Considerations

### 1. Conversation Termination

For real agent scenarios, we can consider various termination strategies:
- **Timeout-based**: No messages for X seconds
- **Pattern-based**: Detect closing phrases ("Have a nice day", "Goodbye", etc.)
- **LLM-based**: Analyze conversation completeness
- **Goal completion**: Check if customer intent was satisfied
- **Outcome-based**: Terminate when a specific outcome is reached

### 2. Intent Handling

- Make intent optional for real agent scenarios
- Implement intent inference from conversation flow
- Support multiple intents per conversation

### 3. Outcome Evaluation

**Centralized approach**: The runner inspects the full conversation and decides when an outcome is reached. This keeps the logic simple and ensures consistent outcome detection across all simulations.

### 4. Message Timing and Conflicts

- Implement realistic typing delays for simulated participants
- Handle overlapping messages gracefully
- Consider message queuing for better realism

### 5. Error Handling

- Channel communication failures
- Participant response timeouts
- Message validation and formatting
- Graceful degradation for partial failures

## Technical Implementation

### 1. LLM Integration
- Use **LangChain** for LLM integration with async chains
- All LLM calls through LangChain should use async methods
- Implement LLM result caching for performance
- Support configurable LLM providers

### 2. Performance Considerations
- Async LLM calls for non-blocking operations
- Caching of LLM calls and embeddings
- Support for parallel chat execution

### 3. Simulation Techniques
- Fine-tuned LLMs for realistic participant behavior
- RAG (Retrieval Augmented Generation) using existing chats as context
- Conversation intent to bootstrap simulations

## Implementation Notes

1. Use asyncio for all I/O operations
2. Implement proper logging for debugging
3. Make components testable in isolation
4. Support conversation persistence
