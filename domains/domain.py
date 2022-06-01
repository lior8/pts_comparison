from abc import ABC, abstractmethod
from dataclasses import dataclass


class Domain(ABC):
    @abstractmethod
    def heuristic(self, state):
        pass

    @abstractmethod
    def goal_test(self, state):
        pass

    @abstractmethod
    def get_successors_and_op_cost(self, state):
        pass


@dataclass(frozen=True, order=True)
class DomainState:
    pass
