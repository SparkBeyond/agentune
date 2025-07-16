import asyncio
from typing import override
from collections.abc import Sequence
import logging

from attrs import define, field, frozen

from agentune.simulate.models.results import ConversationResult
from agentune.simulate.models.scenario import Scenario

from datetime import timedelta

_logger = logging.getLogger(__name__)

class ProgressCallback:
    '''Callbacks that let you monitor the progress of a simulation session.
    
    This class is not abstract to let you override only some methods, or use an instance
    of the base class to request no callbacks.
    '''

    def on_generated_scenarios(self, scenarios: Sequence[Scenario]) -> None: pass

    def on_scenario_start(self, scenario: Scenario) -> None: pass

    def on_scenario_complete(self, scenario: Scenario, result: ConversationResult) -> None: pass

    def on_scenario_failed(self, scenario: Scenario, exception: Exception) -> None: pass

    def on_all_scenarios_complete(self) -> None: pass

@frozen
class ProgressCallbacks(ProgressCallback):
    '''Combines several callback instances into one.'''
    
    callbacks: tuple[ProgressCallback, ...]

    @override
    def on_generated_scenarios(self, scenarios: Sequence[Scenario]) -> None: 
        for callback in self.callbacks:
            callback.on_generated_scenarios(scenarios)

    @override
    def on_scenario_start(self, scenario: Scenario) -> None: 
        for callback in self.callbacks:
            callback.on_scenario_start(scenario)

    @override
    def on_scenario_complete(self, scenario: Scenario, result: ConversationResult) -> None: 
        for callback in self.callbacks:
            callback.on_scenario_complete(scenario, result)

    @override
    def on_scenario_failed(self, scenario: Scenario, exception: Exception) -> None: 
        for callback in self.callbacks:
            callback.on_scenario_failed(scenario, exception)

    @override
    def on_all_scenarios_complete(self) -> None: 
        for callback in self.callbacks:
            callback.on_all_scenarios_complete()

@define
class LoggingProgressCallback(ProgressCallback):
    '''Occasionally logs progress to the module _logger, if it hasn't changed since last time.
    
    Not threadsafe; meant for the callback methods to be called on the same thread
    as the async method.
    '''
    log_interval: timedelta
    
    done: bool = field(init=False, default=False)
    total: int | None = field(init=False, default=None)
    completed: int = field(init=False, default=0)
    _last_logged_count: int | None = field(init=False, default=None)
    
    async def log_progress(self) -> None:
        '''Runs until the task is cancelled, or all scenarios are complete.'''
        while not self.done:
            await asyncio.sleep(self.log_interval.total_seconds())
            if self.total is not None and self._last_logged_count != self.completed:
                _logger.info(f"Progress: {self.completed}/{self.total} scenarios completed")
                self._last_logged_count = self.completed
    
    @override
    def on_generated_scenarios(self, scenarios: Sequence[Scenario]) -> None: 
        self.total = len(scenarios)

    def on_scenario_complete(self, scenario: Scenario, result: ConversationResult) -> None: 
        self.completed += 1

    def on_scenario_failed(self, scenario: Scenario, exception: Exception) -> None: 
        self.completed += 1

    def on_all_scenarios_complete(self) -> None: 
        self.done = True

