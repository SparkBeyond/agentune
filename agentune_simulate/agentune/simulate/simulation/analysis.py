"""Analysis functionality for simulation results."""

from collections import Counter
import random
from collections.abc import Iterable
import logging

from agentune.simulate.simulation.adversarial.base import AdversarialTest

from .. import Outcomes, Scenario, Outcome
from ..models.conversation import Conversation
from ..outcome_detection.base import OutcomeDetectionTest, OutcomeDetector

from ..models.results import SimulatedConversation, OriginalConversation, SimulationAnalysisResult
from ..models.analysis import (
    OutcomeDistribution,
    OutcomeDistributionComparison,
    MessageDistributionStats,
    MessageDistributionComparison,
    AdversarialEvaluationResult,
)
from .adversarial import AdversarialTester

_logger = logging.getLogger(__name__)


async def analyze_simulation_results(
    original_conversations: tuple[OriginalConversation, ...],
    simulated_conversations: tuple[SimulatedConversation, ...],
    adversarial_tester: AdversarialTester,
    outcome_detector: OutcomeDetector,
    scenarios: tuple[Scenario, ...],
    outcomes: Outcomes,
    return_exceptions: bool = True,
) -> SimulationAnalysisResult:
    """Analyze simulation results and generate comprehensive comparison.
    
    Args:
        original_conversations: Real conversations used as input
        simulated_conversations: Generated conversations from simulation
        adversarial_tester: Adversarial tester for evaluation
        outcome_detector: Detector for outcome prediction
        scenarios: Scenarios used for generating conversations
        outcomes: Legal outcome labels for the simulation run
        return_exceptions: If False, raises an error if any per-conversation task raises an error.
                           If True, discards such conversations from the result.
                           (This method's return type does not allow it to actually return the exceptions.)

    Returns:
        Complete analysis result with all comparisons
    """
    # Extract just the conversation objects for analysis
    original_convs = [oc.conversation for oc in original_conversations]  # These are the original conversations, without ids
    simulated_convs = [sc.conversation for sc in simulated_conversations]

    message_comparison = _analyze_message_distributions(
        original_convs, simulated_convs
    )

    # Create a mapping from original_conversation_id to intent from scenarios
    conversation_id_to_intent = {
        scenario.original_conversation_id: scenario.intent
        for scenario in scenarios
    }

    # Generate outcome comparison between the original conversations GT and their predicted outcomes
    conversations_for_outcome_prediction = [original_conv for original_conv in original_conversations if original_conv.id in conversation_id_to_intent]
    original_conversations_predicted_outcomes = await outcome_detector.detect_outcomes(
        tuple(OutcomeDetectionTest(original_conv.conversation, conversation_id_to_intent[original_conv.id]) for original_conv in conversations_for_outcome_prediction),
        possible_outcomes=outcomes,
        return_exceptions=return_exceptions
    )

    for predicted_outcome in original_conversations_predicted_outcomes:
        if isinstance(predicted_outcome, Exception):
            _logger.error('Error trying to predict outcome', exc_info=predicted_outcome)

    def apply_outcome_if_defined(conversation: Conversation, outcome: Outcome | None) -> Conversation:
        """Apply the predicted outcome to the conversation if it is defined."""
        if outcome is not None:
            return conversation.set_outcome(outcome=outcome)
        return conversation

    # Only set outcomes for conversations where we got a valid prediction
    original_conversations_with_predicted_outcomes = [
        apply_outcome_if_defined(original_conversation.conversation, predicted_outcome)
        for original_conversation, predicted_outcome in zip(conversations_for_outcome_prediction, original_conversations_predicted_outcomes)
        if not isinstance(predicted_outcome, Exception)
    ]

    # Perform all analysis
    outcome_comparison = _analyze_outcome_distributions(
        original_convs, simulated_convs, original_conversations_with_predicted_outcomes
    )

    adversarial_evaluation = await _evaluate_adversarial_quality(
        original_convs, simulated_convs, adversarial_tester, return_exceptions=return_exceptions,
    )

    return SimulationAnalysisResult(
        outcome_comparison=outcome_comparison,
        message_distribution_comparison=message_comparison,
        adversarial_evaluation=adversarial_evaluation,
    )


def _outcome_distribution(conversations: Iterable[Conversation]) -> OutcomeDistribution:
    """Compute an ``OutcomeDistribution`` for a collection of conversations."""
    counts: Counter[str] = Counter()
    no_outcome = 0

    for conv in conversations:
        if conv.outcome:
            counts[conv.outcome.name] += 1
        else:
            no_outcome += 1

    total_conversations = counts.total() + no_outcome

    # Convert the Counter to a sorted dictionary to ensure consistent ordering
    sorted_counts = dict(sorted(counts.items()))

    return OutcomeDistribution(
        total_conversations=total_conversations,
        outcome_counts=sorted_counts,
        conversations_without_outcome=no_outcome,
    )


def _analyze_outcome_distributions(
    original_conversations: list[Conversation],
    simulated_conversations: list[Conversation],
    original_conversations_with_predicted_outcomes: list[Conversation],
) -> OutcomeDistributionComparison:
    """Analyze and compare outcome distributions for real vs. generated conversations."""

    original_dist = _outcome_distribution(original_conversations)
    simulated_dist = _outcome_distribution(simulated_conversations)
    original_with_predicted_outcomes_dist = _outcome_distribution(original_conversations_with_predicted_outcomes)

    return OutcomeDistributionComparison(
        original_distribution=original_dist,
        simulated_distribution=simulated_dist,
        original_with_predicted_outcomes=original_with_predicted_outcomes_dist,
    )


def _analyze_message_distributions(
    original_conversations: list[Conversation],
    simulated_conversations: list[Conversation],
) -> MessageDistributionComparison:
    """Analyze and compare message count distributions.
    
    Args:
        original_conversations: Real conversations  
        simulated_conversations: Generated conversations
        
    Returns:
        Comparison of message count distributions
    """
    def _compute_stats(conversations: list[Conversation]) -> MessageDistributionStats:
        if not conversations:
            return MessageDistributionStats(
                min_messages=0,
                max_messages=0,
                mean_messages=0.0,
                median_messages=0.0,
                std_dev_messages=0.0,
                message_count_distribution={},
            )

        message_counts = [len(conv.messages) for conv in conversations]
        message_counts.sort()

        # Basic statistics
        min_msgs = min(message_counts)
        max_msgs = max(message_counts)
        mean_msgs = sum(message_counts) / len(message_counts)
        median_msgs = float(message_counts[len(message_counts) // 2])

        # Standard deviation
        variance = sum((x - mean_msgs) ** 2 for x in message_counts) / len(message_counts)
        std_dev = variance ** 0.5

        # Distribution
        distribution: dict[int, int] = {}
        for count in message_counts:
            distribution[count] = distribution.get(count, 0) + 1

        return MessageDistributionStats(
            min_messages=min_msgs,
            max_messages=max_msgs,
            mean_messages=mean_msgs,
            median_messages=median_msgs,
            std_dev_messages=std_dev,
            message_count_distribution=distribution,
        )

    original_stats = _compute_stats(original_conversations)
    simulated_stats = _compute_stats(simulated_conversations)

    return MessageDistributionComparison(
        original_stats=original_stats,
        simulated_stats=simulated_stats,
    )


def _sample_conversation_pairs(
    original_conversations: list[Conversation],
    simulated_conversations: list[Conversation],
    max_pairs: int,
) -> tuple[list[Conversation], list[Conversation]]:
    """Prepare batches of real and simulated conversations for evaluation.

    If `max_pairs` is specified, it randomly samples pairs. Otherwise, it
    creates pairs from the full Cartesian product.

    Args:
        original_conversations: A list of real conversations.
        simulated_conversations: A list of simulated conversations.
        max_pairs: The maximum number of pairs to randomly sample.

    Returns:
        A tuple containing two lists: the real conversation batch and the
        simulated conversation batch.
    """
    real_batch = []
    simulated_batch = []

    num_originals = len(original_conversations)
    num_simulated = len(simulated_conversations)
    total_possible_pairs = num_originals * num_simulated

    use_all_pairs = max_pairs >= total_possible_pairs

    if use_all_pairs:
        for o_conv in original_conversations:
            for s_conv in simulated_conversations:
                real_batch.append(o_conv)
                simulated_batch.append(s_conv)
    else:
        # Randomly sample unique indices from the flattened space of all pairs
        sampled_indices = random.sample(range(total_possible_pairs), k=max_pairs)

        for index in sampled_indices:
            # Convert the flat index back to a 2D index (original, simulated)
            original_idx, sim_idx = divmod(index, num_simulated)
            real_batch.append(original_conversations[original_idx])
            simulated_batch.append(simulated_conversations[sim_idx])

    return real_batch, simulated_batch


async def _evaluate_adversarial_quality(
    original_conversations: list[Conversation],
    simulated_conversations: list[Conversation],
    adversarial_tester: AdversarialTester,
    max_pairs: int = 100,
    return_exceptions: bool = True,
) -> AdversarialEvaluationResult:
    """Evaluate simulation quality using adversarial testing across a random sample of conversation pairs.
    
    This function orchestrates the adversarial evaluation by preparing conversation
    pairs (either all or a random sample) and using a tester to identify
    the real ones.
    
    Args:
        original_conversations: Real conversations
        simulated_conversations: Generated conversations  
        adversarial_tester: Tester to distinguish real vs simulated
        max_pairs: The maximum number of pairs to randomly sample

    Returns:
        Adversarial evaluation results with accuracy metrics
    """
    if not original_conversations or not simulated_conversations:
        return AdversarialEvaluationResult(0, 0)

    if not adversarial_tester.get_examples():
        example_conversations = random.sample(original_conversations, k=3)
        adversarial_tester = adversarial_tester._with_examples(example_conversations)
        original_conversations = [conv for conv in original_conversations if conv not in example_conversations]
    else:
        original_conversations = [conv for conv in original_conversations if conv not in adversarial_tester.get_examples()]

    # Delegate pair selection logic to the helper function
    real_batch, simulated_batch = _sample_conversation_pairs(
        original_conversations,
        simulated_conversations,
        max_pairs,
    )
    adversarial_tests = tuple(AdversarialTest(real_conv, simulated_conv) for real_conv, simulated_conv in zip(real_batch, simulated_batch))

    if not real_batch:
        return AdversarialEvaluationResult(0, 0)

    results = await adversarial_tester.identify_real_conversations(
        adversarial_tests,
        return_exceptions=return_exceptions
    )

    for result in results:
        if isinstance(result, Exception):
            _logger.error('Error trying to identify real conversation', exc_info=result)

    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    total_evaluated = len(valid_results)

    if total_evaluated == 0:
        return AdversarialEvaluationResult(0, 0)

    correct = sum(1 for result in valid_results if result)

    return AdversarialEvaluationResult(
        total_pairs_evaluated=total_evaluated,
        correct_identifications=correct,
    )