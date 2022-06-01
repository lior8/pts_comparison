import heapq
import time
from dataclasses import dataclass, field

from tqdm import tqdm

from domains.domain import DomainState, Domain
from domains.pancakes import PancakesState, Pancakes
from domains.tile_puzzle import TilePuzzle, TilePuzzleState


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


class Timeout(Exception):
    pass


class NoSolution(Exception):
    pass


class PotentialSearch:
    expanded: int
    generated: int
    reopened: int
    cost: float
    cost_lower_bound: float
    total_time: float

    def __init__(self, domain: Domain):
        self._domain = domain
        self._reset_stats()

    def _reset_stats(self):
        self.expanded = 0
        self.generated = 0
        self.reopened = 0
        self.cost = None
        self.total_time = None

    def solve(self, init_state, c, pure_heuristic_search=False, timeout=60, quiet=False):
        # If we are dealing with pure heuristic search f(n)=h(n), in the case of potential search f(n)=u(n)
        # Which means the used formula (smaller is better)
        def calc_priority(g, h):
            if pure_heuristic_search:
                return h
            else:
                return h / (c - g)

        start_time = time.time()
        root_h = self._domain.heuristic(init_state)
        root = SearchNode(calc_priority(0, root_h), root_h, 0, init_state)
        self.generated += 1
        # This might be somewhat counter-intuitive, but we update closed alongside open, since we cannot search in
        # O(1) in a priority queue, that is why we have an in_open field in SearchNode
        closed = {root.state: root}
        open_ = [root]
        with tqdm(disable=quiet) as pbar:  # Progress bar (helps to see search speed)
            while open_:
                # Check for timeouts
                if time.time() - start_time > timeout:
                    raise Timeout(f"Timed out after {time.time() - start_time} seconds.")

                node = heapq.heappop(open_)
                node.in_open = False

                # If the node is not valid, that means we updated its f(n). Since updating a priority queue is
                # expensive, instead of updating the node, we insert an updated node as a new node, and set the previous
                # to be invalid. Then we encounter an invalid node, we simply discard it
                if not node.is_valid:
                    continue

                self.expanded += 1
                pbar.update(1)

                # Iterate over the neighbors
                for (neighbor, cost_to) in self._domain.get_successors_and_op_cost(node.state):
                    g_neighbor = node.g + cost_to

                    # If the node already exists (i.e. we saw it before, and  it is in open or closed (checked only in
                    # closed since closed holds all nodes) and its g is bigger than the one we've seen, we discard it
                    # because we have a cheaper way to get to that node
                    if neighbor in closed and closed[neighbor].g <= g_neighbor:
                        continue

                    h_neighbor = self._domain.heuristic(neighbor)
                    # If the f(n) of the node is larger or equal to the cost bound, we discard it.
                    if g_neighbor + h_neighbor >= c:
                        continue

                    # Check if it's the goal, and we already know the path cost is under the cost bound
                    if self._domain.goal_test(neighbor):
                        self.total_time = time.time() - start_time
                        self.cost = g_neighbor
                        return self.cost, self.total_time

                    # If it not the goal, that means we are going to either insert a new node into open, or "update" an
                    # already existing node
                    new_node = SearchNode(calc_priority(g_neighbor, h_neighbor), h_neighbor, g_neighbor, neighbor)
                    self.generated += 1

                    # If we have already seen this node before
                    if neighbor in closed:
                        # Is it in open, and we need to update it, or was it already expanded?
                        if closed[neighbor].in_open:
                            # Invalidate the current node in open. Closed always points to the latest (and only valid)
                            # node of a specific state. Also, if the current node in open is better, we would have not
                            # reached here.
                            closed[neighbor].is_valid = False
                        else:
                            self.reopened += 1
                    # We reach here whether the node was in closed or not, and so update the closed dict and push the
                    # node into open
                    closed[neighbor] = new_node
                    heapq.heappush(open_, new_node)

        self.total_time = time.time() - start_time
        raise NoSolution(f"No solution within bound {c}. Elapsed time: {self.total_time} seconds.")


def tile_puzzle_main():
    puzzlestr = '4 7 6 5 10 0 1 13 14 2 15 8 9 3 11 12'
    source = TilePuzzleState(tuple(int(item) for item in puzzlestr.split()))
    domain = TilePuzzle(4, 4)
    pts = PotentialSearch(domain)
    print(pts.solve(source, 60, timeout=3600))


if __name__ == '__main__':
    tile_puzzle_main()
