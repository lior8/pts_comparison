import random
from dataclasses import dataclass
from typing import Tuple

from domains.domain import Domain, DomainState


@dataclass(frozen=True, order=True)
class PancakesState(DomainState):
    """Represents the state of an N tall pancake stack."""
    stack: Tuple[int, ...]  # An N tall pancake stack (wish I had one)


class Pancakes(Domain):
    def __init__(self,
                 ignore_pancakes_up_to: float = 0,
                 size=0,
                 initial_state: PancakesState = None):
        if initial_state is not None:
            self.goal_state = PancakesState(stack=tuple(sorted(initial_state.stack, reverse=True)))
            self.size = len(initial_state.stack)
        elif size > 0:
            self.goal_state = PancakesState(stack=tuple(range(size, 0, -1)))
            self.size = size
        else:
            raise Exception('Must provide initial state or stack size')

        self.half_gap = False
        self.ignore_pancakes_up_to = 0
        self.set_heuristic_degradation(ignore_pancakes_up_to)

    def set_heuristic_degradation(self, ignore_pancakes_up_to):
        if ignore_pancakes_up_to == int(ignore_pancakes_up_to):
            self.ignore_pancakes_up_to = int(ignore_pancakes_up_to)
            self.half_gap = False
        else:
            if ignore_pancakes_up_to - int(ignore_pancakes_up_to) == 0.5:
                self.ignore_pancakes_up_to = int(ignore_pancakes_up_to) + 1
                self.half_gap = True
            else:
                raise Exception("If fraction is given, it has to be 0.5 for half gaps")

    def heuristic(self, state):
        # In this formula, we always add 1 if the max pancake is not at the bottom.
        gaps = 0
        for i in range(len(state.stack) - 1):
            # First we check if there is a gap, and if the right pancake is not ignored (must be true for half gaps as
            # well
            if abs(state.stack[i] - state.stack[i + 1]) > 1 and abs(state.stack[i] - state.stack[i + 1]) > 1:
                # Then we check if the left one is also not ignored or if it is the pancake of the half gap, which means
                # that its gap to its right counts
                if state.stack[i] > self.ignore_pancakes_up_to or\
                        (self.half_gap and state.stack[i] == self.ignore_pancakes_up_to):
                    gaps += 1
        return gaps + (state.stack[0] != max(state.stack))

    def goal_test(self, state):
        return state.stack == self.goal_state.stack

    def get_successors_and_op_cost(self, state):
        return [(PancakesState(state.stack[0:i] + state.stack[i:self.size][::-1]), 1) for i in range(self.size - 1)]

    def generate_instances(self, num_instances, min_ops, max_ops):
        instances = []
        for instance_num in range(num_instances):
            num_ops = random.randrange(min_ops, max_ops + 1)
            stack = list(range(1, self.size + 1))  # Start from the solved state
            stack.reverse()
            for i in range(num_ops):
                flip_i = random.randrange(0, self.size - 1)
                stack = stack[0:flip_i] + stack[flip_i:self.size][::-1]
            instances.append(PancakesState(tuple(stack)))
        return instances
