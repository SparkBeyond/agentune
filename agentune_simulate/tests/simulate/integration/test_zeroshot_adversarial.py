"""Integration tests for the ZeroShotAdversarialTester."""
from datetime import datetime

import pytest
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.fake import FakeListChatModel

from agentune.simulate.models import Conversation, Message
from agentune.simulate.models.roles import ParticipantRole
from agentune.simulate.simulation.adversarial.base import AdversarialTest
from agentune.simulate.simulation.adversarial.zeroshot import ZeroShotAdversarialTester
from agentune.simulate.simulation.analysis import _evaluate_adversarial_quality
from agentune.simulate.models.analysis import AdversarialEvaluationResult


def create_example_conversation() -> Conversation:
    """Create a hardcoded version of the first conversation from the example dataset."""
    customer = ParticipantRole.CUSTOMER
    agent = ParticipantRole.AGENT

    return Conversation(
        messages=(
            Message(
                sender=customer,
                content="Hi I need help",
                timestamp=datetime(2024, 1, 15, 9, 0, 0),
            ),
            Message(
                sender=agent,
                content="Hello, how can I assist you today?",
                timestamp=datetime(2024, 1, 15, 9, 2, 0),
            ),
            Message(
                sender=customer,
                content="I'm having issues with my VPN connection; it keeps dropping and I can't stay connected.",
                timestamp=datetime(2024, 1, 15, 9, 5, 0),
            ),
            Message(
                sender=agent,
                content="Let's start by checking your internet connection. Can you try disconnecting from the VPN and then running a speed test to see if your internet is stable?",
                timestamp=datetime(2024, 1, 15, 9, 9, 0),
            ),
            Message(
                sender=customer,
                content="I've disconnected from the VPN and ran the speed test; my internet connection seems stable with good speeds. What's next?",
                timestamp=datetime(2024, 1, 15, 9, 14, 0),
            ),
            Message(
                sender=agent,
                content="Thank you for checking that. It seems like the issue might be with the VPN settings or the server you're connecting to. Can you try changing the VPN server location and see if that helps?",
                timestamp=datetime(2024, 1, 15, 9, 16, 0),
            ),
        )
    )


@pytest.fixture
def test_conversations() -> tuple[Conversation, Conversation]:
    """Create a pair of conversations for testing. One is real (from example), one is simulated."""
    # Use the first conversation from examples as our real conversation
    real_conversation = create_example_conversation()

    # Create a simple simulated conversation for comparison
    customer = ParticipantRole.CUSTOMER
    agent = ParticipantRole.AGENT
    simulated_conversation = Conversation(
        messages=(
            Message(
                sender=customer,
                content="I want to cancel my service.",
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
async def test_identify_real_conversation_integration(openai_model: ChatOpenAI,
                                                      test_conversations: tuple[Conversation, Conversation]):
    """Test that the tester correctly identifies the real conversation."""
    real_conversation, simulated_conversation = test_conversations
    tester = ZeroShotAdversarialTester(model=openai_model)

    # When we pass the real conversation first, we expect True
    result = await tester.identify_real_conversation(
        AdversarialTest(real_conversation, simulated_conversation)
    )

    # The result should be True since the first conversation is the real one
    assert result is True, "Expected True when first conversation is real"

    # Also test with the conversations swapped - should return False
    swapped_result = await tester.identify_real_conversation(
        AdversarialTest(simulated_conversation, real_conversation)
    )
    assert swapped_result is False, "Expected False when second conversation is real"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_identify_real_conversations_batch_integration(openai_model: ChatOpenAI,
                                                             test_conversations: tuple[Conversation, Conversation]):
    """Test batch processing with a real LLM."""
    real_conv, sim_conv = test_conversations

    # Create a list of conversation pairs
    real_convs = [real_conv, sim_conv]  # Test with swapped roles too
    sim_convs = [sim_conv, real_conv]
    instances = tuple(AdversarialTest(r, s) for r, s in zip(real_convs, sim_convs))

    tester = ZeroShotAdversarialTester(model=openai_model, max_concurrency=2)
    results = await tester.identify_real_conversations(instances)

    assert isinstance(results, tuple)
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
    result1 = await tester.identify_real_conversation(AdversarialTest(empty_conversation, real_conversation))
    assert result1 is None

    # Test with empty second conversation
    result2 = await tester.identify_real_conversation(AdversarialTest(empty_conversation, real_conversation))
    assert result2 is None

    # Test batch with empty conversation
    results = await tester.identify_real_conversations((
        AdversarialTest(empty_conversation, real_conversation),
        AdversarialTest(real_conversation, empty_conversation)
    ))
    assert results == (None, None)


@pytest.mark.asyncio
async def test_extract_label_behavior(openai_model):
    """Test that _extract_label correctly validates and returns conversation labels."""
    tester = ZeroShotAdversarialTester(model=openai_model)

    # Test valid labels
    assert tester._extract_label({"real_conversation": "A"}) == "A"
    assert tester._extract_label({"real_conversation": "B"}) == "B"

    # Test invalid labels
    assert tester._extract_label({"real_conversation": "X"}) is None
    assert tester._extract_label({"real_conversation": ""}) is None
    assert tester._extract_label({"real_conversation": None}) is None
    assert tester._extract_label({}) is None

    # Test invalid types
    assert tester._extract_label({"real_conversation": 123}) is None
    assert tester._extract_label({"real_conversation": ["A"]}) is None


@pytest.mark.asyncio
async def test_evaluate_adversarial_quality_integration(openai_model: ChatOpenAI,
                                                        test_conversations: tuple[Conversation, Conversation]):
    """Test the end-to-end evaluation of conversation quality using the adversarial tester."""
    real_conv1, sim_conv = test_conversations

    # Create a second and third distinct real conversation
    real_conv2 = Conversation(messages=(
        Message(sender=ParticipantRole.CUSTOMER, content="This is a second real conversation about a different topic.",
                timestamp=datetime(2024, 1, 2, 10, 0)),
        Message(sender=ParticipantRole.AGENT, content="Understood. I can help with that.",
                timestamp=datetime(2024, 1, 2, 10, 1)),
    ))
    real_conv3 = Conversation(messages=(
        Message(sender=ParticipantRole.CUSTOMER, content="And a third one, just to be sure.",
                timestamp=datetime(2024, 1, 3, 10, 0)),
        Message(sender=ParticipantRole.AGENT, content="Thank you for providing these examples.",
                timestamp=datetime(2024, 1, 3, 10, 1)),
    ))

    # Create multiple test conversations, with one designated as the example
    original_conversations = [real_conv1, real_conv2, real_conv3]
    simulated_conversations = [sim_conv, sim_conv, sim_conv]
    example_conversations = (real_conv1,)

    # Create the tester, providing one example.
    # The evaluation function will filter this one out, leaving 2 for evaluation.
    tester = ZeroShotAdversarialTester(model=openai_model, example_conversations=example_conversations)

    # Run the evaluation
    result = await _evaluate_adversarial_quality(
        original_conversations=original_conversations,
        simulated_conversations=simulated_conversations,
        adversarial_tester=tester
    )

    # Verify the result structure
    assert isinstance(result, AdversarialEvaluationResult)
    # After filtering 1 example, 2 real are left. 2 real * 3 simulated = 6 pairs.
    assert result.total_pairs_evaluated == 6, f"Expected 6 pairs (2x3 combinations), got {result.total_pairs_evaluated}"
    # We can't predict exact accuracy, but it should be between 0 and 1 (inclusive)
    assert 0 <= result.accuracy <= 1.0


@pytest.mark.asyncio
async def test_empty_conversations_in_batch():
    """Test that the adversarial tester correctly handles various conversation combinations in a batch."""
    # Create more distinct real conversations with natural variations and imperfections
    real_conv1 = Conversation(messages=(
        Message(sender=ParticipantRole.CUSTOMER,
                content="Hi, my order #12345 hasn't arrived yet and it's 3 days late. Can you check?",
                timestamp=datetime(2024, 1, 1, 10, 0)),
        Message(sender=ParticipantRole.AGENT,
                content="I apologize for the delay. Let me check the status of order #12345 for you. One moment please...",
                timestamp=datetime(2024, 1, 1, 10, 2)),
        Message(sender=ParticipantRole.AGENT,
                content="I see the issue. There was a delay at our warehouse. Your order is now in transit and should arrive by Friday.",
                timestamp=datetime(2024, 1, 1, 10, 4)),
    ))

    real_conv2 = Conversation(messages=(
        Message(sender=ParticipantRole.CUSTOMER,
                content="I was charged twice for my subscription this month. Can you fix this?",
                timestamp=datetime(2024, 1, 1, 11, 0)),
        Message(sender=ParticipantRole.AGENT,
                content="I'm sorry to hear about the duplicate charge. Let me look into this for you.",
                timestamp=datetime(2024, 1, 1, 11, 1)),
        Message(sender=ParticipantRole.AGENT,
                content="I've processed a refund for the duplicate charge. It should appear in your account within 5-7 business days.",
                timestamp=datetime(2024, 1, 1, 11, 3)),
    ))

    # Create more obviously simulated conversations with different patterns
    sim_conv1 = Conversation(messages=(
        Message(sender=ParticipantRole.CUSTOMER,
                content="Greetings. I am experiencing an issue with my recent purchase.",
                timestamp=datetime(2024, 1, 1, 10, 0)),
        Message(sender=ParticipantRole.AGENT,
                content="Hello valued customer! We appreciate your business. Could you please provide more details about your concern?",
                timestamp=datetime(2024, 1, 1, 10, 1)),
    ))

    # Make the second simulated conversation even more distinct with different patterns
    sim_conv2 = Conversation(messages=(
        Message(sender=ParticipantRole.CUSTOMER, content="I require immediate assistance with a critical issue",
                timestamp=datetime(2024, 1, 1, 10, 0)),
        Message(sender=ParticipantRole.AGENT,
                content="Dear customer, we apologize for the inconvenience. Our team is here to assist you with your concern.",
                timestamp=datetime(2024, 1, 1, 10, 1)),
        Message(sender=ParticipantRole.AGENT,
                content="Please rest assured that we are working diligently to resolve this matter for you.",
                timestamp=datetime(2024, 1, 1, 10, 3)),
    ))

    empty_conversation = Conversation(messages=())

    # Create a FakeListChatModel that will return predetermined responses
    # These responses simulate the LLM choosing which conversation is real
    # The responses should be the raw content strings that would be parsed by the output parser
    fake_responses = [
        '{"real_conversation": "A"}',  # First response
        '{"real_conversation": "B"}',  # Second response
        '{"real_conversation": "A"}',  # Third response
    ]

    # Create the fake chat model with our predetermined responses
    model = FakeListChatModel(responses=fake_responses)

    # Use a fixed random seed for reproducibility
    # Note: The actual order of A/B is random, but will be consistent with the same seed
    tester = ZeroShotAdversarialTester(model=model, random_seed=42)

    # Create batches with various combinations
    real_convs = [real_conv1, real_conv1, real_conv2, empty_conversation]
    sim_convs = [sim_conv1, sim_conv2, sim_conv1, sim_conv2]
    instances = tuple(AdversarialTest(r, s) for r, s in zip(real_convs, sim_convs))

    # Run the batch test
    batch_results = await tester.identify_real_conversations(instances)

    # Verify the results
    assert len(batch_results) == 4, f"Expected 4 results, got {len(batch_results)}"

    # With random_seed=42, we know the exact results to expect
    # These specific assertions work because the random seed is fixed
    # The expected values were determined by running the test with this seed
    assert batch_results[0] is True, f"Expected True for first pair, got {batch_results[0]}"
    assert batch_results[1] is False, f"Expected False for second pair, got {batch_results[1]}"
    assert batch_results[2] is False, f"Expected False for third pair, got {batch_results[2]}"

    # Fourth pair should be None (empty conversation)
    assert batch_results[3] is None, "Expected None for empty conversation pair"
