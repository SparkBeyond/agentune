"""Analysis functionality for simulation results."""

from collections import Counter
from typing import Iterable
import asyncio

from .. import Outcomes, Scenario
from ..models.conversation import Conversation
from ..outcome_detection.base import OutcomeDetector

from ..models.results import SimulatedConversation, OriginalConversation, SimulationAnalysisResult
from ..models.analysis import (
    OutcomeDistribution,
    OutcomeDistributionComparison,
    MessageDistributionStats,
    MessageDistributionComparison,
    AdversarialEvaluationResult,
)
from .adversarial import AdversarialTester


async def analyze_simulation_results(
    original_conversations: tuple[OriginalConversation, ...],
    simulated_conversations: tuple[SimulatedConversation, ...],
    adversarial_tester: AdversarialTester,
    outcome_detector: OutcomeDetector,
    scenarios: tuple[Scenario, ...],
    outcomes: Outcomes,
) -> SimulationAnalysisResult:
    """Analyze simulation results and generate comprehensive comparison.
    
    Args:
        original_conversations: Real conversations used as input
        simulated_conversations: Generated conversations from simulation
        adversarial_tester: Adversarial tester for evaluation
        outcome_detector: Detector for outcome prediction
        scenarios: Scenarios used for generating conversations
        outcomes: Legal outcome labels for the simulation run
        
    Returns:
        Complete analysis result with all comparisons
    """
    # Extract just the conversation objects for analysis
    original_convs = [oc.conversation for oc in original_conversations]  # These are the original conversations, without ids
    simulated_convs = [sc.conversation for sc in simulated_conversations]

    message_comparison = _analyze_message_distributions(
        original_convs, simulated_convs
    )
    adversarial_evaluation = await _evaluate_adversarial_quality(
        original_convs, simulated_convs, adversarial_tester
    )

    # Create a mapping from original_conversation_id to intent from scenarios
    conversation_id_to_intent = {
        scenario.original_conversation_id: scenario.intent
        for scenario in scenarios
        if scenario.original_conversation_id is not None
    }

    # Generate outcome comparison between the original conversations GT and their predicted outcomes
    original_conversations_with_predicted_outcome_tasks = []
    conversations_for_outcome_prediction = []

    for conv in original_conversations:
        if conv.id in conversation_id_to_intent:
            intent = conversation_id_to_intent[conv.id]
            original_conversations_with_predicted_outcome_tasks.append(
                outcome_detector.detect_outcome(conv.conversation, intent, possible_outcomes=outcomes)
            )
            conversations_for_outcome_prediction.append(conv.conversation)


    original_conversations_predicted_outcomes = await asyncio.gather(
        *original_conversations_with_predicted_outcome_tasks
    )

    # Only set outcomes for conversations where we got a valid prediction
    original_conversations_with_predicted_outcomes = [
        conv.set_outcome(outcome=predicted_outcome)
        for conv, predicted_outcome in zip(conversations_for_outcome_prediction, original_conversations_predicted_outcomes)
        if predicted_outcome is not None
    ]

    # Perform all analysis
    outcome_comparison = _analyze_outcome_distributions(
        original_convs, simulated_convs, original_conversations_with_predicted_outcomes
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


async def _evaluate_adversarial_quality(
    original_conversations: list[Conversation],
    simulated_conversations: list[Conversation],
    adversarial_tester: AdversarialTester,
) -> AdversarialEvaluationResult:
    """Evaluate simulation quality using adversarial testing.
    
    Args:
        original_conversations: Real conversations
        simulated_conversations: Generated conversations  
        adversarial_tester: Tester to distinguish real vs simulated
        
    Returns:
        Adversarial evaluation results with accuracy metrics
    """
    # todo: Implement adversarial evaluation logic
    
    return AdversarialEvaluationResult(
        total_pairs_evaluated=-1,
        correct_identifications=-1,
    )
