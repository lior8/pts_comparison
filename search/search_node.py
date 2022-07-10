from dataclasses import field, dataclass

from domains.domain import DomainState


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
