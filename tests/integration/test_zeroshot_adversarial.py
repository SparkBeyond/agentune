"""Integration tests for the ZeroShotAdversarialTester."""
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from langchain_openai import ChatOpenAI

from conversation_simulator.models import Conversation, Message
from conversation_simulator.models.roles import ParticipantRole
from conversation_simulator.simulation.adversarial import ZeroShotAdversarialTester


def create_dch2_conversation() -> Conversation:
    """Create a hardcoded version of the first conversation from the dch2 dataset."""
    customer = ParticipantRole.CUSTOMER
    agent = ParticipantRole.AGENT
    
    return Conversation(
        messages=(
            Message(
                sender=customer,
                content=("Last night, I waited in line for 2 hours in the business office, but because I only had a copy of my ID card and didn't bring the original, "
                        "I was not allowed to cancel the broadband service, and I had to charge for suspending the service! I brought the original ID with me "
                        "according to the reservation tonight, but the store manager actually said that the set-top box should be returned to cancel it or 500 "
                        "yuan deposit should be paid first. Many restrictions have been imposed on customers to cancel their business, and you have not yet "
                        "made it clear to customers. We need to come to the store for so many times! Is it fun to play with consumers? @ China Telecom Guangdong "
                        "Customer Service Guangzhou·Jingxi"),
                timestamp=datetime(2024, 1, 15, 9, 0, 0),
            ),
            Message(
                sender=agent,
                content=("We're very sorry, I am the Guangdong Customer Service Staff of China Telecom. I have paid attention to your feedback. We will continue to improve "
                        "our service to satisfy our customers. Please continue to supervise. Thank you."),
                timestamp=datetime(2024, 1, 15, 9, 2, 0),
            ),
            Message(
                sender=customer,
                content="How can consumers supervise you if you don't solve your own problems?",
                timestamp=datetime(2024, 1, 15, 9, 5, 0),
            ),
            Message(
                sender=agent,
                content=("We will continue to improve various services and improve our service quality. Thank you for your suggestion."),
                timestamp=datetime(2024, 1, 15, 9, 9, 0),
            ),
            Message(
                sender=customer,
                content=("Nonsense. China Telecom has failed to make progress for so many years. It's simply a national shame. No wonder more and more people have decided "
                        "never to use you again!"),
                timestamp=datetime(2024, 1, 15, 9, 14, 0),
            ),
            Message(
                sender=agent,
                content=("We're really sorry for the inconvenience. I suggest that you can register feedback through online complaints+consultation and "
                        "complaints-self-service-China Telecom Huango website· Guangdong. After registration, the processing specialist will carefully check "
                        "it and reply to you. Thank you."),
                timestamp=datetime(2024, 1, 15, 9, 16, 0),
            ),
        )
    )


@pytest.fixture
def test_conversations() -> tuple[Conversation, Conversation]:
    """Create a pair of conversations for testing. One is real (from dch2), one is simulated."""
    # Use the first conversation from dch2 as our real conversation
    real_conversation = create_dch2_conversation()
    
    # Create a simple simulated conversation for comparison
    customer = ParticipantRole.CUSTOMER
    agent = ParticipantRole.AGENT
    simulated_conversation = Conversation(
        messages=(
            Message(
                sender=customer,
                content="I want to cancel my broadband service.",
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
            ),
            Message(
                sender=agent,
                content="Please provide your account details.",
                timestamp=datetime(2024, 1, 1, 10, 1, 0),
            ),
            Message(
                sender=customer,
                content="Why is it so complicated to cancel?",
                timestamp=datetime(2024, 1, 1, 10, 2, 0),
            ),
            Message(
                sender=agent,
                content="I'm sorry for the inconvenience. Let me help you with that.",
                timestamp=datetime(2024, 1, 1, 10, 3, 0),
            ),
        )
    )
    
    return real_conversation, simulated_conversation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_identify_real_conversation_integration(openai_model: ChatOpenAI, test_conversations: tuple[Conversation, Conversation]):
    """Test that the tester returns a boolean result with a real LLM."""
    real_conversation, simulated_conversation = test_conversations
    tester = ZeroShotAdversarialTester(model=openai_model)

    # We expect this to be True or False, but we'll just check the type to avoid flaky tests
    result = await tester.identify_real_conversation(
        real_conversation, simulated_conversation
    )

    # The result should be a boolean (True/False) or None if there was an error
    assert result is None or isinstance(result, bool)


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
    # Results can be boolean or None
    assert all(res is None or isinstance(res, bool) for res in results)


@pytest.mark.asyncio
async def test_identify_real_conversation_returns_none_for_empty(openai_model: ChatOpenAI):
    """Test that empty conversations return None."""
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

    tester = ZeroShotAdversarialTester(model=openai_model)

    # Test with empty first conversation
    result1 = await tester.identify_real_conversation(empty_conversation, real_conversation)
    assert result1 is None

    # Test with empty second conversation
    result2 = await tester.identify_real_conversation(real_conversation, empty_conversation)
    assert result2 is None

    # Test batch with empty conversation
    results = await tester.identify_real_conversations(
        [empty_conversation, real_conversation],
        [real_conversation, empty_conversation]
    )
    assert results == [None, None]


@pytest.mark.asyncio
async def test_identify_real_conversation_invalid_response(openai_model: ChatOpenAI, test_conversations: tuple[Conversation, Conversation]):
    """Test that invalid LLM responses log a warning and return None."""
    real_conv, sim_conv = test_conversations
    
    # Mock the chain to return an invalid response
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = {"real_conversation": "X"}  # Invalid response
    mock_chain.abatch.return_value = [{"real_conversation": "X"}]
    
    with patch.object(ZeroShotAdversarialTester, '_create_adversarial_chain', return_value=mock_chain), \
         patch('conversation_simulator.simulation.adversarial.zeroshot.logger.warning') as mock_warning:
        tester = ZeroShotAdversarialTester(model=openai_model)
        
        # Test single conversation - should return None for invalid response
        result = await tester.identify_real_conversation(real_conv, sim_conv)
        assert result is None
        mock_warning.assert_called()
        
        # Clear the mock for the next test
        mock_warning.reset_mock()
        
        # Test batch - should also return None for invalid response
        results = await tester.identify_real_conversations([real_conv], [sim_conv])
        assert results == [None]  # Should be None due to invalid response
        mock_warning.assert_called()


@pytest.mark.asyncio
async def test_response_order_consistency(openai_model: ChatOpenAI, test_conversations: tuple[Conversation, Conversation]):
    """Test that the order of responses is consistent with the same random seed."""
    real_conv, sim_conv = test_conversations
    
    # Create a mock chain that always returns a specific label
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = {"real_conversation": "A"}
    mock_chain.abatch.return_value = [{"real_conversation": "A"}, {"real_conversation": "A"}]
    
    with patch.object(ZeroShotAdversarialTester, '_create_adversarial_chain', return_value=mock_chain):
        # Create two testers with the same random seed
        tester1 = ZeroShotAdversarialTester(model=openai_model, random_seed=42)
        tester2 = ZeroShotAdversarialTester(model=openai_model, random_seed=42)
        
        # Run the same test twice with the same seed
        result1 = await tester1.identify_real_conversation(real_conv, sim_conv)
        result2 = await tester2.identify_real_conversation(real_conv, sim_conv)
        
        # Results should be consistent with the same seed
        assert result1 == result2
        
        # Now test with a different random seed
        tester3 = ZeroShotAdversarialTester(model=openai_model, random_seed=43)
        # We don't assert anything about this result, just demonstrating different seeds
        _ = await tester3.identify_real_conversation(real_conv, sim_conv)
        
        # The result might be different with a different seed due to different ordering
        # But we can't assert that it will always be different, as it's probabilistic
        
        # Test batch processing with the same seed
        batch_results1 = await tester1.identify_real_conversations(
            [real_conv, real_conv], [sim_conv, sim_conv]
        )
        batch_results2 = await tester2.identify_real_conversations(
            [real_conv, real_conv], [sim_conv, sim_conv]
        )
        
        # Batch results should also be consistent with the same seed
        assert batch_results1 == batch_results2
