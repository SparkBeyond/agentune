"""Tests for RichConsoleReporter formatting and tree building."""
from datetime import timedelta

import pytest

from agentune.core.progress.base import ProgressStage
from agentune.core.progress.reporters.rich_console import RichConsoleReporter


@pytest.fixture
def reporter_with_colors() -> RichConsoleReporter:
    return RichConsoleReporter(timedelta(seconds=0.1), show_percentages=True, show_colors=True)


@pytest.fixture
def reporter_no_colors() -> RichConsoleReporter:
    return RichConsoleReporter(timedelta(seconds=0.1), show_percentages=True, show_colors=False)


@pytest.fixture
def reporter_no_percentages() -> RichConsoleReporter:
    return RichConsoleReporter(timedelta(seconds=0.1), show_percentages=False, show_colors=True)


def test_label_in_progress_with_colors(reporter_with_colors: RichConsoleReporter) -> None:
    """In-progress stage with count shows cyan name."""
    stage = ProgressStage(name='task', count=5, total=10)
    label = reporter_with_colors._format_stage_label(stage)
    
    assert '[cyan]task[/cyan]' in label
    assert '5/10' in label
    assert '50.0%' in label


def test_label_completed_with_colors(reporter_with_colors: RichConsoleReporter) -> None:
    """Completed stage shows green name and checkmark."""
    stage = ProgressStage(name='done', count=10, total=10)
    stage.complete()
    label = reporter_with_colors._format_stage_label(stage)
    
    assert '[green]done[/green]' in label
    assert '[green]✓[/green]' in label


def test_label_started_no_count_with_colors(reporter_with_colors: RichConsoleReporter) -> None:
    """Stage without count shows white name."""
    stage = ProgressStage(name='waiting')
    label = reporter_with_colors._format_stage_label(stage)
    
    assert '[white]waiting[/white]' in label


def test_label_without_colors(reporter_no_colors: RichConsoleReporter) -> None:
    """Without colors, labels have no Rich markup."""
    stage = ProgressStage(name='task', count=5, total=10)
    label = reporter_no_colors._format_stage_label(stage)
    
    assert '[' not in label or label.startswith('task [')  # Only progress brackets, no color markup
    assert 'task' in label
    assert '5/10' in label


def test_label_completed_without_colors(reporter_no_colors: RichConsoleReporter) -> None:
    """Completed stage without colors shows plain checkmark."""
    stage = ProgressStage(name='done')
    stage.complete()
    label = reporter_no_colors._format_stage_label(stage)
    
    assert label == 'done ✓'


def test_percentage_display(reporter_with_colors: RichConsoleReporter) -> None:
    """Percentage shown when show_percentages=True."""
    stage = ProgressStage(name='work', count=3, total=12)
    label = reporter_with_colors._format_stage_label(stage)
    
    assert '25.0%' in label


def test_no_percentage_display(reporter_no_percentages: RichConsoleReporter) -> None:
    """Percentage hidden when show_percentages=False."""
    stage = ProgressStage(name='work', count=3, total=12)
    label = reporter_no_percentages._format_stage_label(stage)
    
    assert '%' not in label
    assert '3/12' in label


def test_count_only_no_total(reporter_with_colors: RichConsoleReporter) -> None:
    """Count without total shows just the count."""
    stage = ProgressStage(name='items', count=42)
    label = reporter_with_colors._format_stage_label(stage)
    
    assert '[42]' in label
    assert '42/' not in label


def test_total_only_no_count(reporter_with_colors: RichConsoleReporter) -> None:
    """Total without count shows 0/total."""
    stage = ProgressStage(name='pending', total=100)
    label = reporter_with_colors._format_stage_label(stage)
    
    assert '[0/100]' in label


def test_tree_builds_hierarchy(reporter_with_colors: RichConsoleReporter) -> None:
    """Tree structure reflects stage hierarchy."""
    root = ProgressStage(name='root')
    root.add_child('child1')
    root.add_child('child2')
    
    tree = reporter_with_colors._build_tree(root)
    
    assert len(tree.children) == 2

