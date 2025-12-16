"""Tests for ProgressStage diff calculations."""
from agentune.core.progress.base import ProgressStage, stage_scope


def test_diff_no_changes() -> None:
    """No diff when nothing changed."""
    with stage_scope('root', count=0, total=10) as stage:
        snapshot1 = stage.deepcopy()
        snapshot2 = stage.deepcopy()
    
    diffs = ProgressStage.diff(snapshot1, snapshot2)
    assert diffs == []


def test_diff_count_changed() -> None:
    """Diff detects count changes."""
    with stage_scope('root', count=0, total=10) as stage:
        snapshot1 = stage.deepcopy()
        stage.set_count(5)
        snapshot2 = stage.deepcopy()
    
    diffs = ProgressStage.diff(snapshot1, snapshot2)
    assert len(diffs) == 1
    assert diffs[0].path == ('root',)
    assert diffs[0].count_changed
    assert diffs[0].old_count == 0
    assert diffs[0].new_count == 5
    assert not diffs[0].completed
    assert not diffs[0].added


def test_diff_total_changed() -> None:
    """Diff detects total changes."""
    with stage_scope('root', count=0) as stage:
        snapshot1 = stage.deepcopy()
        stage.set_total(100)
        snapshot2 = stage.deepcopy()
    
    diffs = ProgressStage.diff(snapshot1, snapshot2)
    assert len(diffs) == 1
    assert diffs[0].total_changed
    assert diffs[0].old_total is None
    assert diffs[0].new_total == 100


def test_diff_child_added() -> None:
    """Diff detects new child stages."""
    with stage_scope('root') as root:
        snapshot1 = root.deepcopy()
        with stage_scope('child', count=0, total=5):
            snapshot2 = root.deepcopy()
    
    diffs = ProgressStage.diff(snapshot1, snapshot2)
    assert len(diffs) == 1
    assert diffs[0].path == ('root', 'child')
    assert diffs[0].added
    assert diffs[0].new_count == 0
    assert diffs[0].new_total == 5


def test_diff_child_completed() -> None:
    """Diff detects child stage completion."""
    with stage_scope('root') as root:
        with stage_scope('child', count=5, total=10) as child:
            snapshot1 = root.deepcopy()
            child.complete()
            snapshot2 = root.deepcopy()
    
    diffs = ProgressStage.diff(snapshot1, snapshot2)
    assert len(diffs) == 1
    assert diffs[0].path == ('root', 'child')
    assert diffs[0].completed
    assert diffs[0].new_total == 5


def test_diff_new_stage_already_completed() -> None:
    """Diff marks new stage as completed if it's already completed."""
    with stage_scope('root') as root:
        snapshot1 = root.deepcopy()
        with stage_scope('child', count=5, total=10):
            pass
        snapshot2 = root.deepcopy()
    
    diffs = ProgressStage.diff(snapshot1, snapshot2)
    assert len(diffs) == 1
    diff = diffs[0]
    assert diff.path == ('root', 'child')
    assert diff.added
    assert diff.completed, "New stage that's already completed should have completed=True"


def test_diff_nested_children() -> None:
    """Diff handles nested child stages correctly."""
    with stage_scope('root') as root:
        snapshot1 = root.deepcopy()
        with stage_scope('level1'):
            with stage_scope('level2', count=0, total=3):
                snapshot2 = root.deepcopy()
    
    diffs = ProgressStage.diff(snapshot1, snapshot2)
    assert len(diffs) == 2
    paths = [d.path for d in diffs]
    assert ('root', 'level1') in paths
    assert ('root', 'level1', 'level2') in paths


def test_diff_from_none() -> None:
    """Diff from None treats all stages as added."""
    with stage_scope('root', count=0, total=10) as root:
        with stage_scope('child', count=1, total=5):
            snapshot = root.deepcopy()
    
    diffs = ProgressStage.diff(None, snapshot)
    assert len(diffs) == 2
    assert all(d.added for d in diffs)

