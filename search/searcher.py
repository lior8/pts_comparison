from abc import ABC, abstractmethod

from domains.domain import Domain


class Searcher(ABC):
    expanded: int
    generated: int
    reopened: int
    cost: float
    cost_lower_bound: float
    total_time: float

    def __init__(self, domain: Domain):
        self._domain = domain
        self.expanded = 0
        self.generated = 0
        self.reopened = 0
        self.cost = None
        self.total_time = None

    def reset_stats(self):
        self.expanded = 0
        self.generated = 0
        self.reopened = 0
        self.cost = None
        self.total_time = None

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass


class Timeout(Exception):
    pass


class NoSolution(Exception):
    pass
