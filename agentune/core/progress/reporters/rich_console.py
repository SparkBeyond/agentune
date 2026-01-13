"""Rich console-based progress reporter for interactive display."""

from __future__ import annotations

from datetime import timedelta
from typing import override

from rich.console import Console
from rich.live import Live
from rich.tree import Tree

from agentune.core.progress.base import ProgressStage, root_stage
from agentune.core.progress.reporters.base import ProgressReporter


class RichConsoleReporter(ProgressReporter):
    """Progress reporter that displays interactive progress in the console using Rich.
    
    Progress updates are displayed in a hierarchical tree visualization. Can be used in interactive terminals and Jupyter notebooks.
    
    Args:
        poll_interval: How often to poll for progress updates.
        show_percentages: Whether to show percentage completion for counted stages.
        show_colors: Whether to use colors in the display.
    """

    def __init__(
        self,
        poll_interval: timedelta,
        show_percentages: bool,
        show_colors: bool,
    ) -> None:
        self.poll_interval = poll_interval
        self._show_percentages = show_percentages
        self._show_colors = show_colors
        self._live: Live | None = None
        self._progress_tree: Tree | None = None

    @override
    async def start(self, root_stage: ProgressStage) -> None:
        """Start displaying progress for the given root stage."""
        snapshot = root_stage.deepcopy()
        self._progress_tree = self._build_tree(snapshot)
        self._live = Live(
            self._progress_tree,
            console=Console(),
            auto_refresh=False,
        )
        self._live.start(refresh=True)

    @override
    async def update(self, snapshot: ProgressStage) -> None:
        """Update the display with the latest progress snapshot."""
        if self._live is None:
            return
        self._progress_tree = self._build_tree(snapshot)
        self._live.update(self._progress_tree, refresh=True)

    @override
    async def stop(self) -> None:
        """Stop displaying and perform cleanup."""
        if self._live is not None:
            current_root = root_stage()
            if current_root is not None:
                await self.update(current_root.deepcopy())
            self._live.stop()
            self._live = None
        self._progress_tree = None

    def _build_tree(self, stage: ProgressStage) -> Tree:
        """Build a Rich Tree structure from a progress stage.
        
        Args:
            stage: The progress stage to convert.
            
        Returns:
            A Rich Tree object representing the progress hierarchy.
        """
        label = self._format_stage_label(stage)
        tree = Tree(label)
        
        for child in stage.children:
            tree.add(self._build_tree(child))
        
        return tree

    def _format_stage_label(self, stage: ProgressStage) -> str:
        """Format a stage label with optional Rich markup for colors."""
        count = stage.count
        total = stage.total
        
        progress = ''
        if count is not None and total is not None:
            if self._show_percentages and total > 0:
                percentage = (count / total) * 100
                progress = f' [{count}/{total} ({percentage:.1f}%)]'
            else:
                progress = f' [{count}/{total}]'
        elif count is not None:
            progress = f' [{count}]'
        elif total is not None:
            progress = f' [0/{total}]'
        
        if not self._show_colors:
            name = stage.name + progress
            return name + ' ✓' if stage.is_completed else name
        
        if stage.is_completed:
            return f'[green]{stage.name}[/green][dim]{progress}[/dim] [green]✓[/green]'
        elif count is not None:
            return f'[cyan]{stage.name}[/cyan][dim]{progress}[/dim]'
        else:
            return f'[white]{stage.name}[/white][dim]{progress}[/dim]'

