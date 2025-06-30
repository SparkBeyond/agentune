from typing import Sequence

from conversation_simulator.models.results import ConversationResult
from conversation_simulator.models.scenario import Scenario

class ProgressCallbacks:
    '''Callbacks that let you monitor the progress of a simulation session.
    
    This class is not abstract to let you override only some methods, or use an instance
    of the base class to request no callbacks.
    '''

    def on_generated_scenarios(self, scenarios: Sequence[Scenario]) -> None: pass

    def on_scenario_start(self, scenario: Scenario) -> None: pass

    def on_scenario_complete(self, scenario: Scenario, result: ConversationResult) -> None: pass

    def on_scenario_failed(self, scenario: Scenario, exception: Exception) -> None: pass

    def on_all_scenarios_complete(self) -> None: pass

