import asyncio
from attrs import define, field, frozen
import pytest
import logging
import random
import functools

from conversation_simulator.util.asyncutil import bounded_parallelism

_logger = logging.getLogger(__name__)

@frozen
class Started:
    task: int

@frozen
class Ended:
    task: int

@frozen
class Failed:
    task: int
    exception: Exception

@define
class BoundedParallelismRunner:
    max_concurrent_tasks: int
    tasks: list[int]
    return_exceptions: bool
    fail_below_threshold: float = 0.0 # If 0.0, tasks never fail

    events: list[Started | Ended | Failed] = field(init=False, factory=list)
    rand: random.Random = field(init=False, factory=lambda: random.Random(42))

    async def run_task(self, task: int) -> int:
        self.events.append(Started(task))
        await asyncio.sleep(self.rand.random() * 0.1)
        if self.rand.random() < self.fail_below_threshold:
            e = Exception(f'Task {task} failed')
            self.events.append(Failed(task, e))
            raise e
        else:
            self.events.append(Ended(task))
            return task
    
    async def run_all_and_assert(self) -> None:
        _logger.info(f'Testing {len(self.tasks)} tasks with {self.max_concurrent_tasks=}, {self.return_exceptions=}, {self.fail_below_threshold=}')
        runnables = [functools.partial(self.run_task, task) for task in self.tasks]
        try:
            results = await bounded_parallelism(runnables, self.max_concurrent_tasks, return_exceptions=self.return_exceptions)
            self.test_results(results)
        except Exception:
            assert self.fail_below_threshold > 0.0 and not self.return_exceptions
    
    def test_results(self, results: list[int | Exception]) -> None:
        assert len(results) == len(self.tasks)

        if self.return_exceptions and self.fail_below_threshold > 0.0:
            assert any(isinstance(result, Exception) for result in results)
            assert any(isinstance(result, int) for result in results)
        elif not self.return_exceptions and self.fail_below_threshold > 0.0:
            # TODO 
            pass
        else:
            assert all(isinstance(result, int) for result in results)

        assert all(x == y for x, y in zip(results, self.tasks) if not isinstance(x, Exception))

        concurrent = 0
        running = set[int]()
        max_started = -1
        for event in self.events:
            if isinstance(event, Started):
                concurrent += 1
                assert concurrent <= self.max_concurrent_tasks, "No more than max_concurrent_tasks should be running at any time"
                assert event.task == max_started + 1, "Tasks started in the order of the input list"
                max_started = event.task
                running.add(event.task)
            elif isinstance(event, Ended) or isinstance(event, Failed):
                assert event.task in running
                running.remove(event.task)
                concurrent -= 1
        assert running == set(), "All tasks completed"

@pytest.mark.asyncio
async def test_bounded_parallelism():
    for max_concurrent_tasks in [1, 2, 10]:
        for tasks_size in list(range(1, 10, 2)) + [50]:
            runner = BoundedParallelismRunner(max_concurrent_tasks, list(range(tasks_size)), False, 0.0)
            await runner.run_all_and_assert()

@pytest.mark.asyncio
async def test_bounded_parallelism_return_exceptions():
    for max_concurrent_tasks in [1, 5]:
        for tasks_size in [30]:
            runner = BoundedParallelismRunner(max_concurrent_tasks, list(range(tasks_size)), True, 0.1)
            await runner.run_all_and_assert()

@pytest.mark.asyncio
async def test_bounded_parallelism_failure_dont_return_exceptions():
    for max_concurrent_tasks in [1, 5]:
        for tasks_size in [30]:
            runner = BoundedParallelismRunner(max_concurrent_tasks, list(range(tasks_size)), False, 0.1)
            await runner.run_all_and_assert()

