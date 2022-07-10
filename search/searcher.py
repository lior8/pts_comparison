from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from domains.domain import Domain, DomainState


@dataclass(unsafe_hash=True, order=True)  # Not frozen to allow changing h and, consequently, f.
# The members that participate in the hash are immutable.
class SearchNode:
    f: float = field(hash=False)  # Order of fields determines comparison order
    h: float = field(hash=False)  # Tie-break in favor of smaller h -> higher g.
    g: float = field(hash=False)
    state: DomainState  # It does participate in __cmp__ (and TilePuzzleState is defined with the default
    # order=True) only to make tie-breaking deterministic
    parent: "SearchNode" = field(default=None, repr=False, hash=False, compare=False)  # The string is a forward
    # reference
    in_open: bool = field(default=True, repr=False, hash=False, compare=False)
    is_valid: bool = field(default=True, repr=False, hash=False, compare=False)


class Searcher(ABC):
    expanded: int
    generated: int
    reopened: int
    cost: float
    cost_lower_bound: float
    total_time: float

    def __init__(self, domain: Domain):
        self.domain = domain
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
