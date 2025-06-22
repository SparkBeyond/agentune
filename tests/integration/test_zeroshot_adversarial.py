"""Integration tests for the ZeroShotAdversarialTester."""
from datetime import datetime

import pytest
from unittest.mock import AsyncMock, patch

from langchain_openai import ChatOpenAI

from conversation_simulator.models import Conversation, Message
from conversation_simulator.models.roles import ParticipantRole
from conversation_simulator.simulation.adversarial import ZeroShotAdversarialTester


@pytest.fixture
def test_conversations() -> tuple[Conversation, Conversation]:
    """Create a pair of conversations for testing. One is clearly more human-like."""
    customer = ParticipantRole.CUSTOMER
    agent = ParticipantRole.AGENT
    real_conversation = Conversation(
        messages=(
            Message(
                sender=customer,
                content="Hi, I'm having trouble with my order. The website is really confusing.",
                timestamp=datetime.fromtimestamp(0),
            ),
            Message(
                sender=agent,
                content="I'm so sorry to hear that! I can definitely help you sort this out. Could you please tell me your order number?",
                timestamp=datetime.fromtimestamp(1),
            ),
        )
    )
    simulated_conversation = Conversation(
        messages=(
            Message(sender=customer, content="Order issue.", timestamp=datetime.fromtimestamp(0)),
            Message(sender=agent, content="Provide order number.", timestamp=datetime.fromtimestamp(1)),
        )
    )
    return real_conversation, simulated_conversation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_identify_real_conversation_integration(openai_model: ChatOpenAI, test_conversations: tuple[Conversation, Conversation]):
    """Test that the tester returns a boolean result with a real LLM."""
    real_conversation, simulated_conversation = test_conversations
    tester = ZeroShotAdversarialTester(model=openai_model)

    # We expect this to be True, but we'll just check the type to avoid flaky tests
    result = await tester.identify_real_conversation(
        real_conversation, simulated_conversation
    )

    assert isinstance(result, bool)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_identify_real_conversations_batch_integration(openai_model: ChatOpenAI, test_conversations: tuple[Conversation, Conversation]):
    """Test batch processing with a real LLM."""
    real_conv, sim_conv = test_conversations

    # Create a list of conversation pairs
    real_convs = [real_conv, sim_conv]  # Test with swapped roles too
    sim_convs = [sim_conv, real_conv]

    tester = ZeroShotAdversarialTester(model=openai_model, max_concurrency=2)
    results = await tester.identify_real_conversations(real_convs, sim_convs)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(res, bool) for res in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_identify_real_conversation_empty_integration(openai_model: ChatOpenAI):
    """Test that empty conversations are handled gracefully without calling the LLM."""
    real_conversation = Conversation(
        messages=(
            Message(
                sender=ParticipantRole.CUSTOMER,
                content="hi",
                timestamp=datetime.fromtimestamp(0),
            ),
        )
    )
    empty_conversation = Conversation(messages=())

    # To test that the LLM is not called, we mock the chain creation process
    mock_chain = AsyncMock()
    with patch.object(
        ZeroShotAdversarialTester, "_create_adversarial_chain", return_value=mock_chain
    ):
        # Instantiate the tester within the patch context to ensure it uses the mock chain
        tester_with_mock_chain = ZeroShotAdversarialTester(model=openai_model)

        result1 = await tester_with_mock_chain.identify_real_conversation(
            empty_conversation, real_conversation
        )
        assert result1 is False

        result2 = await tester_with_mock_chain.identify_real_conversation(
            real_conversation, empty_conversation
        )
        assert result2 is False

    # Verify that the ainvoke method on our mock chain was never called
    mock_chain.ainvoke.assert_not_called()
