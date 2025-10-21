from __future__ import annotations

import contextlib
import contextvars
import datetime
import itertools
import threading
from collections.abc import Generator


class ProgressStage:
    """A node publishing progress information, with a name, optional count and total, start date and completion information.

    Stages are organized into a tree which corresponds to the call tree and is managed by a context var;
    they should be created by functions in this module such as stage_scope(). Creating a class instance directly
    won't do anything because it won't be published for anyone else to read.

    Progress reporters can call current_stage() or root_stage() and then the deepcopy() method to get a snapshot
    of the current progress state and display it.

    Stages are mutable and threadsafe. The allowed updates for a stage are:
     - Add child stage
     - Update count and/or total (while keeping total >= count)
     - Mark as completed. This marks all descendant stages as completed, sets total = count and disallows future updates,
       including adding more children.

    See docs/progress.md for more details.
    """

    __slots__ = ('_children', '_completed', '_count', '_lock', '_name', '_parent', '_started', '_total')

    def __init__(self, name: str, count: int | None = None, total: int | None = None,
                 parent: ProgressStage | None = None) -> None:
        if count is not None and total is not None and total < count:
            raise ValueError(f'total {total} < count {count}')
        if parent is self:
            raise ValueError('Cannot create a stage with itself as parent')

        self._name = name
        self._count = count
        self._total = total
        self._parent = parent
        self._started = datetime.datetime.now()
        self._completed: datetime.datetime | None = None
        self._children: tuple[ProgressStage, ...] = ()
        self._lock = threading.Lock()

    @property
    def name(self) -> str: return self._name
    @property
    def count(self) -> int | None: return self._count
    @property
    def total(self) -> int | None: return self._total
    @property
    def started(self) -> datetime.datetime: return self._started
    @property
    def completed(self) -> datetime.datetime | None: return self._completed
    @property
    def parent(self) -> ProgressStage | None: return self._parent
    @property
    def children(self) -> tuple[ProgressStage, ...]: return self._children

    @property
    def is_completed(self) -> bool:
        return self._completed is not None

    def __str__(self) -> str:
        snapshot = self.deepcopy()
        base = snapshot.name
        if snapshot.count is not None or snapshot.total is not None:
            base += f' ({snapshot.count}/{snapshot.total})'
        if snapshot.is_completed:
            base += ' [completed]'
        for child in snapshot.children:
            base += f'\n\\-> {str(child).replace('\\n', '   \\n')}'
        return base


    @property
    def root(self) -> ProgressStage:
        if self.parent is None:
            return self
        return self.parent.root

    def set_count(self, value: int) -> None:
        with self._lock:
            if self.is_completed:
                raise ValueError('Cannot update count after stage is completed')
            if self.total is not None and value > self.total:
                raise ValueError(f'Cannot update count to {value} > total {self.total}')
            self._count = value

    def increment_count(self, value: int) -> None:
        """Increment self.count by value; if self.count is None, set it to value."""
        with self._lock:
            if self.is_completed:
                raise ValueError('Cannot update count after stage is completed')
            if self._count is None:
                self._count = 0
            if self._total is not None and self._count + value > self._total:
                raise ValueError(f'Cannot update count to {self._count + value} > total {self._total}')
            self._count += value

    def set_total(self, value: int) -> None:
        with self._lock:
            if self.is_completed:
                raise ValueError('Cannot update count after stage is completed')
            if self.count is not None and value < self.count:
                raise ValueError(f'Cannot update total to {value} < count {self.count}')
            self._total = value


    def complete(self) -> None:
        """Set `completed` to the current time and disallow future updates; if count is not None, set total=count.

        If any children are not yet completed, complete them (recursively) before completing this stage.
        """
        with self._lock:
            if self.is_completed:
                pass
            if self.count is not None:
                self._total = self.count
            for child in self.children:
                child.complete()
            self._completed = datetime.datetime.now()


    def add_child(self, name: str, count: int | None = None, total: int | None = None) -> ProgressStage:
        """Create a new stage as a child of this stage and return it."""
        child = ProgressStage(name, count, total, parent=self)
        with self._lock:
            if self.is_completed:
                raise ValueError('Cannot add child to completed stage')
            if any(stage.name == name for stage in self.children):
                raise ValueError(f'Child stage with name {name} already exists')
            self._children += (child,)
            return child

    def deepcopy(self) -> ProgressStage:
        """Return a consistent, valid copy of this (sub)tree."""
        with self._lock:
            copy = ProgressStage(self.name, self.count, self.total, parent=self.parent)
            copy._started = self.started
            copy._completed = self.completed
            copy._children = tuple(child.deepcopy() for child in self.children)
            return copy

_progress = contextvars.ContextVar[ProgressStage | None]('progress_stage', default=None)

def current_stage() -> ProgressStage | None:
    return _progress.get()

def root_stage() -> ProgressStage | None:
    stage = current_stage()
    return stage.root if stage is not None else None

@contextlib.contextmanager
def stage_scope(name: str, count: int | None = None, total: int | None = None,
                unique_name: bool = True) -> Generator[ProgressStage]:
    """Create a new stage as a child of the current stage and return it, as a context manager.

    If there is no current stage, creates a new root stage.
    The stage is completed when exiting the context.

    Args:
        unique_name: if True, and a stage already exists with this name as a child of the current stage, add a
                     unique numeric suffix to make the new stage name valid.
    """
    match current_stage():
        case None:
            stage = ProgressStage(name, count, total)
            _progress.set(stage)
            try:
                yield stage
            finally:
                stage.complete()
                _progress.set(None)
        case stage:
            if unique_name:
                existing_names = {child.name for child in stage.children}
                if name in existing_names:
                    names = (f'{name} ({i})' for i in itertools.count())
                    name = next(name for name in names if name not in existing_names)

            child = stage.add_child(name, count, total)
            token = _progress.set(child)
            try:
                yield child
            finally:
                child.complete()
                _progress.reset(token)

@contextlib.contextmanager
def root_stage_scope(name: str, count: int | None = None, total: int | None = None) -> Generator[ProgressStage]:
    """As stage_scope, but always creates a new root stage.

    If there is an existing stage, the new stage will not be a child of it.
    Typical code should NOT use this; it is very rarely necessary to force a new root stage.
    """
    stage = ProgressStage(name, count, total)
    token = _progress.set(stage)
    try:
        yield stage
    finally:
        stage.complete()
        _progress.reset(token)
