"""Analysis functionality for simulation results."""

from collections import Counter
import random
from typing import Iterable

from ..models.conversation import Conversation
from ..models.results import SimulatedConversation, SimulationAnalysisResult
from ..models.analysis import (
    OutcomeDistribution,
    OutcomeDistributionComparison,
    MessageDistributionStats,
    MessageDistributionComparison,
    AdversarialEvaluationResult,
)
from .adversarial import AdversarialTester


async def analyze_simulation_results(
    original_conversations: tuple[Conversation, ...],
    simulated_conversations: tuple[SimulatedConversation, ...],
    adversarial_tester: AdversarialTester,
) -> SimulationAnalysisResult:
    """Analyze simulation results and generate comprehensive comparison.
    
    Args:
        original_conversations: Real conversations used as input
        simulated_conversations: Generated conversations from simulation
        adversarial_tester: Adversarial tester for evaluation
        
    Returns:
        Complete analysis result with all comparisons
    """
    # Extract just the conversation objects for analysis
    simulated_convs = [sc.conversation for sc in simulated_conversations]
    
    # Perform all analysis
    outcome_comparison = _analyze_outcome_distributions(
        list(original_conversations), simulated_convs
    )
    message_comparison = _analyze_message_distributions(
        list(original_conversations), simulated_convs
    )
    adversarial_evaluation = await _evaluate_adversarial_quality(
        list(original_conversations), simulated_convs, adversarial_tester
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
    return OutcomeDistribution(
        total_conversations=total_conversations,
        outcome_counts=dict(counts),
        conversations_without_outcome=no_outcome,
    )


def _analyze_outcome_distributions(
    original_conversations: list[Conversation],
    simulated_conversations: list[Conversation],
) -> OutcomeDistributionComparison:
    """Analyze and compare outcome distributions for real vs. generated conversations."""

    original_dist = _outcome_distribution(original_conversations)
    simulated_dist = _outcome_distribution(simulated_conversations)

    return OutcomeDistributionComparison(
        original_distribution=original_dist,
        simulated_distribution=simulated_dist,
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
    max_pairs: int = 20,
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
    max_pairs: int = 20,
) -> AdversarialEvaluationResult:
    """Evaluate simulation quality using adversarial testing across all combinations.
    
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

    # Delegate pair selection logic to the helper function
    real_batch, simulated_batch = _sample_conversation_pairs(
        original_conversations,
        simulated_conversations,
        max_pairs,
    )

    if not real_batch:
        return AdversarialEvaluationResult(0, 0)

    results = await adversarial_tester.identify_real_conversations(
        tuple(real_batch),
        tuple(simulated_batch)
    )
    
    valid_results = [r for r in results if r is not None]
    total_evaluated = len(valid_results)
    
    if total_evaluated == 0:
        return AdversarialEvaluationResult(0, 0)
        
    correct = sum(1 for result in valid_results if result)
    
    return AdversarialEvaluationResult(
        total_pairs_evaluated=total_evaluated,
        correct_identifications=correct,
    )