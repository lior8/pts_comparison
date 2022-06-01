import random
from dataclasses import field, dataclass
from enum import Enum
from typing import Tuple, List

from domains.domain import Domain, DomainState


@dataclass(frozen=True, order=True)
class TilePuzzleState(DomainState):
    """Represents the state of an MxN tile puzzle."""

    puzzle: Tuple[int, ...]  # An MxN-sized list
    blank: int = field(default=-1, hash=False, compare=False)

    def __post_init__(self):
        if self.blank == -1:
            object.__setattr__(self, "blank", self.puzzle.index(0))


class SlideDirection(Enum):
    left = 0
    up = 1
    down = 2
    right = 3


class TilePuzzle(Domain):
    def __init__(
            self,
            width, height,
            operator_order=(
                    SlideDirection.right,
                    SlideDirection.left,
                    SlideDirection.down,
                    SlideDirection.up,
            ),
            goal_state=None,
            ignore_tiles_up_to=0
    ):
        self.h_increment = None
        self.goal_state = goal_state
        self.width = width
        self.height = height
        self.size = self.width * self.height
        self.ignore_tiles_up_to = ignore_tiles_up_to

        if self.ignore_tiles_up_to < 0:
            raise Exception("Only positive number of tiles can be ignored")

        if set(operator_order) != set(SlideDirection):
            raise Exception("All 4 operators need to be provided to specify their order")
        self.operators_in_order = operator_order

        self.applicable_operators = [[]] * self.size
        for blank in range(self.size):
            applicable = []
            for op in self.operators_in_order:
                if op == SlideDirection.up and blank > self.width - 1:
                    applicable.append(op)
                elif op == SlideDirection.left and blank % self.width > 0:
                    applicable.append(op)
                elif op == SlideDirection.right and blank % self.width < self.width - 1:
                    applicable.append(op)
                elif op == SlideDirection.down and blank < self.width * self.height - self.width:
                    applicable.append(op)
            self.applicable_operators[blank] = applicable

        if goal_state is None:
            goal_state = TilePuzzleState(tuple(range(self.size)), 0)
        self.set_goal(goal_state)

    def set_goal(self, goal_state):
        if set(goal_state.puzzle) != set(range(self.size)):
            raise Exception(f"Bad goal state {goal_state}")

        self.goal_state = goal_state

        self.h_increment = [None] * self.size
        for i in range(1, self.size):
            self.h_increment[i] = [0] * self.size
        for goal_pos in range(self.size):
            tile = goal_state.puzzle[goal_pos]
            if tile == 0:  # Blank doesn't contribute to h because it's moved in every operator
                # and we count the movement of the other tile each time
                continue
            for pos in range(self.size):
                self.h_increment[tile][pos] = abs(goal_pos % self.width - pos % self.width) + abs(
                    goal_pos // self.width - pos // self.width
                )  # # difference in column + difference in row

    def heuristic(self, state) -> int:
        min_dist = 0

        for i, tile in enumerate(state.puzzle):
            if tile <= self.ignore_tiles_up_to:
                continue
            min_dist += self.h_increment[tile][i]

        return min_dist

    def goal_test(self, state):
        return self.goal_state == state

    def get_successors_and_op_cost(self, state: TilePuzzleState) -> List[Tuple[TilePuzzleState, int]]:
        neighbors_and_op_costs = []
        for op in self.applicable_operators[state.blank]:
            new_state_puzzle, new_state_blank = self.apply_op(op, state.puzzle, state.blank)
            neighbors_and_op_costs.append((TilePuzzleState(new_state_puzzle, new_state_blank), 1))

        return neighbors_and_op_costs

    def apply_op(self, op: SlideDirection, orig_puzzle: Tuple[int, ...], blank: int) -> Tuple[Tuple[int, ...], int]:
        #  We actually do the swap to maintain consistency when using abstract states
        #  (these contain -1 in some positions, including possibly the blank position.)
        puzzle = list(orig_puzzle)  # timeit showed copying and swapping is faster than constructing
        # a tuple with slice unpacking and two swapped items. Changing the variable name to help mypy
        # cope with the type change.
        w = self.width
        h = self.height
        if op == SlideDirection.up:
            if blank >= w:
                puzzle[blank], puzzle[blank - w] = puzzle[blank - w], puzzle[blank]
                blank -= w
            else:
                raise Exception(f"Up operator is invalid for {puzzle}")
        elif op == SlideDirection.down:
            if blank < self.size - w:
                puzzle[blank], puzzle[blank + w] = puzzle[blank + w], puzzle[blank]
                blank += w
            else:
                raise Exception(f"Down operator is invalid for {puzzle}")

        elif op == SlideDirection.right:
            if blank % w < w - 1:
                puzzle[blank], puzzle[blank + 1] = puzzle[blank + 1], puzzle[blank]
                blank += 1
            else:
                raise Exception(f"Right operator is invalid for {puzzle}")
        elif op == SlideDirection.left:
            if blank % w > 0:
                puzzle[blank], puzzle[blank - 1] = puzzle[blank - 1], puzzle[blank]
                blank -= 1
            else:
                raise Exception(f"Left operator is invalid for {puzzle}")

        return tuple(puzzle), blank

    def generate_instances(self, num_instances, min_ops, max_ops):
        instances = []
        for instance_num in range(num_instances):
            num_ops = random.randrange(min_ops, max_ops + 1)
            puzzle = tuple(range(self.size))  # Start from the solved state
            blank = 0
            for i in range(num_ops):
                op = random.choice(self.applicable_operators[blank])
                puzzle, blank = self.apply_op(op, puzzle, blank)
            instances.append(TilePuzzleState(puzzle, blank))
        return instances
