
from __future__ import annotations
from datetime import datetime, timedelta

import attrs
from conversation_simulator import Conversation, Message, ParticipantRole
from conversation_simulator.models.message import MessageDraft
from conversation_simulator.participants import Participant


@attrs.frozen
class MessageWithTimestamp:
    """A message with a timestamp, used to represent a message for a participant in a conversation."""
    content: str
    timestamp: datetime


@attrs.frozen
class ConversationSplits:
    """Split messages in a conversation into customer and agent messages.

    initial_message is separated from customer_messages and agent_messages, because it's
    used to init the conversation.  

    customer_messages and agent_messages are tuples of messages, where customer_messages
    contains messages sent by the customer and agent_messages contains messages sent by
    the agent. Both don't include initial_message.

    """
    initial_message: Message | None
    customer_messages: tuple[MessageWithTimestamp | None, ...]
    agent_messages: tuple[MessageWithTimestamp | None, ...]

    def __attrs_post_init__(self) -> None:
        """Validate that if initial_message is None, customer_messages and agent_messages are empty."""
        if self.initial_message is None:
            assert not self.customer_messages
            assert not self.agent_messages

    @staticmethod
    def reconstruct(conversation: Conversation) -> ConversationSplits:
        """Split messages in a conversation into customer and agent messages.

    For any given conversation it's possible to recreate a sequence of messages by
    alternating between customer and agent messages.
    It's not enough to just split messages by sender, because we need to be aware of 
    empty messages - where user returns None, which may result in 2 consecutive messages by the same participant.
    """
   
        # Handle empty conversation
        if not conversation.messages:
            return ConversationSplits(
                initial_message=None,
                customer_messages=(),
                agent_messages=()
            )
        
        # Extract initial message and remaining messages
        initial_message = conversation.messages[0]
        remaining_messages = list(conversation.messages[1:])
        
        # Initialize turn tracking
        customer_turns: list[MessageWithTimestamp | None] = []
        agent_turns: list[MessageWithTimestamp | None] = []

        # Determine who continues the conversation, based on the initial message sender
        current_turn_is_agent = initial_message.sender == ParticipantRole.CUSTOMER
        message_idx = 0
        
        # Continue processing turns until we run out of messages and both participants pass
        while True:
            if message_idx < len(remaining_messages):
                current_message = remaining_messages[message_idx]
                
                if current_turn_is_agent:
                    if current_message.sender == ParticipantRole.AGENT:
                        # Agent's turn and agent message - they spoke
                        agent_turns.append(MessageWithTimestamp(
                            content=current_message.content,
                            timestamp=current_message.timestamp
                        ))
                        message_idx += 1
                    else:
                        # Agent's turn but customer message - agent passed, customer will speak next turn
                        agent_turns.append(None)
                else:
                    if current_message.sender == ParticipantRole.CUSTOMER:
                        # Customer's turn and customer message - they spoke
                        customer_turns.append(MessageWithTimestamp(
                            content=current_message.content,
                            timestamp=current_message.timestamp
                        ))
                        message_idx += 1
                    else:
                        # Customer's turn but agent message - customer passed, agent will speak next turn
                        customer_turns.append(None)
                
                # Switch turns
                current_turn_is_agent = not current_turn_is_agent
                
            else:
                # No more messages - both participants will pass
                agent_turns.append(None)
                customer_turns.append(None)

                # If both participants passed, we can stop processing
                break
        
        return ConversationSplits(
            initial_message=initial_message,
            customer_messages=tuple(customer_turns),
            agent_messages=tuple(agent_turns)
        )


def from_drafts_to_conversation(drafts: tuple[MessageDraft, ...]) -> Conversation:
    """Convert a list of draft messages into a Conversation object.
    
    Create timestamps for each draft message based on the current time, 
    plus 5 sec for each subsequent message.
    """
    current_time = datetime.now()
    messages = tuple(
        Message(
            content=draft.content,
            sender=draft.sender,
            timestamp=current_time + timedelta(seconds=i * 5)
        ) for i, draft in enumerate(drafts)
    )           

    return Conversation(messages=messages)


@attrs.define
class MockTurnBasedParticipant(Participant):
    """Mock participant for turn-based testing that returns or skips messages in sequence.

    Attributes:
        role: The role of this participant
        messages: List of messages to return in sequence (None = finished)
    """

    role: ParticipantRole
    messages: tuple[MessageWithTimestamp | None, ...]
    message_index: int = 0

    def with_intent(self, intent_description: str) -> MockTurnBasedParticipant:
        return self  # Intent is not used in this mock

    async def get_next_message(self, conversation: Conversation) -> Message | None:
        """Return the next message in sequence or None if no more messages or explicitly pass."""
        if self.message_index >= len(self.messages):
            raise ValueError("No more messages available for this participant")

        message = self.messages[self.message_index]
        self.message_index += 1

        # If message is None, this represents a deliberate pass
        if message is None:
            return None

        return Message(
            content=message.content,
            timestamp=message.timestamp,
            sender=self.role
        )
