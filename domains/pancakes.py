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
                 ignore_pancakes_up_to=0,
                 size=0,
                 initial_state: PancakesState = None):
        if initial_state is not None:
            self.goal_state = PancakesState(stack=tuple(sorted(initial_state.stack, reverse=True)))
            self.size = len(initial_state.stack)
        elif size > 0:
            self.goal_state = PancakesState(stack=tuple(range(size - 1, -1, -1)))
            self.size = size
        else:
            raise Exception('Must provide initial state or stack size')
        # If we want to ignore 0 pancakes, we want to include all, but if we want to ignore 2, we
        # want to start the loop from 3, therefore, if it's bigger than 0, we add 1
        self.start_count_from = ignore_pancakes_up_to + (ignore_pancakes_up_to > 0)

    def heuristic(self, state):
        # In this formula, we always add 1 if the max pancake is not at the bottom.

        # Consult someone how to treat this case.
        gaps = sum(abs(state.stack[i] - state.stack[i + 1]) > 1 for i in
                   range(self.start_count_from, self.size - 1))
        return gaps + (state.stack[0] != max(state.stack))

    def goal_test(self, state):
        return state.stack == self.goal_state.stack

    def get_successors_and_op_cost(self, state):
        return [(PancakesState(state.stack[0:i] + state.stack[i:self.size][::-1]), 1) for i in range(self.size - 1)]

    def generate_instances(self, num_instances, min_ops, max_ops):
        instances = []
        for instance_num in range(num_instances):
            num_ops = random.randrange(min_ops, max_ops + 1)
            stack = list(range(self.size))  # Start from the solved state
            stack.reverse()
            for i in range(num_ops):
                flip_i = random.randrange(0, self.size - 1)
                stack = stack[0:flip_i] + stack[flip_i:self.size][::-1]
            instances.append(PancakesState(tuple(stack)))
        return instances


def main():
    domain = Pancakes(size=20)
    d = domain.generate_instances(10, 1, 1)
    for i in d:
        instance_str = ' '.join(str(pancake) for pancake in i.stack)


if __name__ == '__main__':
    main()
