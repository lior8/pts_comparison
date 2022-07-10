import heapq
import time

from tqdm import tqdm

from search.searcher import Searcher, SearchNode, Timeout, NoSolution


class AstarSearcher(Searcher):
    def solve(self, init_state, timeout=60, quiet=False):
        self.reset_stats()
        start_time = time.time()
        root_h = self.domain.heuristic(init_state)
        root = SearchNode(root_h, root_h, 0, init_state)
        self.generated += 1
        # This might be somewhat counter-intuitive, but we update closed alongside open, since we cannot search in
        # O(1) in a priority queue, that is why we have an in_open field in SearchNode
        closed = {root.state: root}
        open_ = [root]
        with tqdm(disable=quiet) as pbar:
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

                if self.domain.goal_test(node.state):
                    self.total_time = time.time() - start_time
                    self.cost = node.g
                    return self.cost, self.total_time

                self.expanded += 1
                pbar.update(1)

                for (neighbor, cost_to) in self.domain.get_successors_and_op_cost(node.state):
                    g_neighbor = node.g + cost_to

                    # If the node already exists (i.e. we saw it before, and  it is in open or closed (checked only in
                    # closed since closed holds all nodes) and its g is bigger than the one we've seen, we discard it
                    # because we have a cheaper way to get to that node
                    if neighbor in closed and closed[neighbor].g <= g_neighbor:
                        continue

                    h_neighbor = self.domain.heuristic(neighbor)

                    # If it not the goal, that means we are going to either insert a new node into open, or "update" an
                    # already existing node
                    new_node = SearchNode(g_neighbor + h_neighbor, h_neighbor, g_neighbor, neighbor)
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
