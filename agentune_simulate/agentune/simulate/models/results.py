"""Result models for simulation outcomes."""

from __future__ import annotations
import json
from datetime import datetime, timedelta
from pathlib import Path
import attrs

from .conversation import Conversation
from .scenario import Scenario
from .analysis import OutcomeDistributionComparison, MessageDistributionComparison, AdversarialEvaluationResult
from ..util.structure import converter


@attrs.frozen
class ConversationResult:
    """Result of simulating a single conversation."""
    
    conversation: Conversation
    duration: timedelta = timedelta(seconds=0)
    
    @property
    def message_count(self) -> int:
        """Number of messages in the conversation."""
        return len(self.conversation.messages)
    
    @property
    def outcome_name(self) -> str | None:
        """Name of the conversation outcome, if any."""
        return self.conversation.outcome.name if self.conversation.outcome else None
    
    def __str__(self) -> str:
        """String representation of the conversation result."""
        outcome_str = f" - {self.outcome_name}" if self.outcome_name else ""
        return (
            f"ConversationResult: {self.message_count} messages, "
            f"{self.duration.total_seconds():.2f}s{outcome_str}"
        )


@attrs.frozen
class OriginalConversation:
    """A real conversation used as input for simulation generation."""
    
    id: str  # Unique identifier for the original conversation
    conversation: Conversation


@attrs.frozen
class SimulatedConversation:
    """A simulated conversation generated from an original conversation."""
    
    id: str  # Unique identifier for this simulated conversation
    scenario_id: str  # ID of the scenario that generated this conversation
    original_conversation_id: str  # Links back to the original
    conversation: Conversation


@attrs.frozen
class SimulationSessionResult:
    """Comprehensive result of a simulation session with analysis capabilities."""
    
    # Session metadata
    session_name: str
    session_description: str
    started_at: datetime
    completed_at: datetime
    
    # Core data
    original_conversations: tuple[OriginalConversation, ...]
    scenarios: tuple[Scenario, ...]
    simulated_conversations: tuple[SimulatedConversation, ...]
    
    # Analysis results
    analysis_result: SimulationAnalysisResult
    
    @property
    def total_original_conversations(self) -> int:
        """Number of original conversations used."""
        return len(self.original_conversations)
    
    @property
    def total_simulated_conversations(self) -> int:
        """Number of simulated conversations generated."""
        return len(self.simulated_conversations)
    
    @property
    def simulation_ratio(self) -> float:
        """Ratio of simulated to original conversations."""
        if self.total_original_conversations == 0:
            return 0.0
        return self.total_simulated_conversations / self.total_original_conversations
    
    def __str__(self) -> str:
        """String representation of the session result."""
        return (
            f"SimulationSessionResult: '{self.session_name}' - "
            f"{self.total_original_conversations} original → {self.total_simulated_conversations} simulated"
        )
    
    def generate_summary(self) -> str:
        """Generate a formatted text summary of the simulation results.
        
        Returns:
            Formatted string with session overview, outcome distribution, and sample conversation
        """
        lines: list = [
            "=" * 40,
            "SIMULATION RESULTS",
            "=" * 40,
            f"Session name: {self.session_name}",
            f"Original conversations: {len(self.original_conversations)}",
            f"Simulated conversations: {len(self.simulated_conversations)}"
        ]

        # Show side-by-side comparison using existing analysis data
        if self.original_conversations and self.simulated_conversations:
            # Message count comparison
            orig_msgs = self.analysis_result.message_distribution_comparison.original_stats.mean_messages
            sim_msgs = self.analysis_result.message_distribution_comparison.simulated_stats.mean_messages
            lines.append("")
            lines.append("Average messages per conversation:")
            lines.append(f"  Original: {orig_msgs:.1f}")
            lines.append(f"  Simulated: {sim_msgs:.1f}")
            
            # Outcome distribution comparison
            orig_dist = self.analysis_result.outcome_comparison.original_distribution
            sim_dist = self.analysis_result.outcome_comparison.simulated_distribution
            
            # Get all unique outcomes from both distributions
            all_outcomes = sorted(set(orig_dist.outcome_counts.keys()) | set(sim_dist.outcome_counts.keys()))
            
            lines.append("")
            lines.append("Outcome distribution comparison:")
            lines.append(f"{'Outcome':<20} {'Original':<15} {'Simulated':<15}")
            lines.append("-" * 50)
            
            for outcome in all_outcomes:
                orig_count = orig_dist.outcome_counts.get(outcome, 0)
                sim_count = sim_dist.outcome_counts.get(outcome, 0)
                orig_pct = orig_dist.outcome_percentages.get(outcome, 0.0)
                sim_pct = sim_dist.outcome_percentages.get(outcome, 0.0)
                
                lines.append(f"{outcome:<20} {orig_count:>3} ({orig_pct:>4.1f}%)   {sim_count:>3} ({sim_pct:>4.1f}%)")
            
            # Show a sample conversation
            if self.simulated_conversations:
                sample_conv = self.simulated_conversations[0].conversation
                lines.append("")
                lines.append(f"Sample conversation ({len(sample_conv.messages)} messages):")
                for i, msg in enumerate(sample_conv.messages[:4]):  # Show first 4 messages
                    content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    lines.append(f"  {i+1}. {msg.sender.value}: {content_preview}")
                if len(sample_conv.messages) > 4:
                    lines.append(f"  ... and {len(sample_conv.messages) - 4} more messages")
        
        lines.append("=" * 40)
        return "\n".join(lines)
    
    def save_to_file(self, output_path: str) -> None:
        """Save the simulation results to a JSON file.
        
        Args:
            output_path: Path where to save the results
        """
        # Convert to dictionary using the structure converter
        result_dict = converter.unstructure(self)
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)


@attrs.frozen
class SimulationAnalysisResult:
    """Wrapper for all simulation analysis results."""
    
    outcome_comparison: OutcomeDistributionComparison
    message_distribution_comparison: MessageDistributionComparison
    adversarial_evaluation: AdversarialEvaluationResult
