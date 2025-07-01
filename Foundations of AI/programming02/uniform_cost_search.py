import search_problem
import heapq
import itertools  # To generate unique tie-breakers

from search import Search, SearchNode


class UniformCostSearch(Search):

    name = "uniform-cost"

    def search(self):
        
        p = self.search_problem

        min_heap = []
        heapq.heapify(min_heap)
        counter = itertools.count()  # Unique tie-breaker generator
        
        heapq.heappush(min_heap, (0, next(counter), SearchNode(p.initial_state, None, 0)))
        self.generated += 1
        reached = set()

        while min_heap:
            _, _, node = heapq.heappop(min_heap)  # Unpack (cost, tie-breaker, node)
            self.expanded += 1

            if node.state not in reached:
                reached.add(node.state)

                if p.is_goal(node.state):
                    return self.extract_path(node), node.g

                for action in p.actions(node.state):
                    succ, cost = p.result(node.state, action)
                    new_g = node.g + cost
                    succ_node = SearchNode(succ, node, new_g)

                    # Add tie-breaker counter to avoid comparing SearchNode objects
                    heapq.heappush(min_heap, (new_g, next(counter), succ_node))
                    self.generated += 1

        # No solution found
        return None, None


if __name__ == "__main__":
  problem = search_problem.generate_random_problem(8, 2, 3, max_cost=10)
  problem.dump()
  ucs = UniformCostSearch(problem, True)
  ucs.run()
  problem.dump("graph.dot")



    #======== Visualization of the generated problem ========#
    # dot -Tpng graph.dot -o digraph.png
    # open digraph.png


    #======== Visualize a newly generated problem ===========#
    # python3
    # from search_problem import generate_random_problem
    # rand = generate_random_problem(8, 2, 3, max_cost=10)
    # rand.dump("graph.dot")
    # exit()
    # dot -Tpng graph.dot -o digraph.png
    # open digraph.png

