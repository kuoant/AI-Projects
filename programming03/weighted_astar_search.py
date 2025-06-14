import pancake_problem

from pancake_problem import PancakeProblem
from queue import PriorityQueue
from search import Search, SearchNode
import heapq
import itertools


class WeightedAStarSearch(Search):
  name = "weighted-astar"

  def __init__(self, search_problem, weight, **kwargs):
    super().__init__(search_problem, **kwargs)
    self.w = weight
    if weight == 0:
      self.name = "uniform-cost"
    elif weight == 1:
      self.name = "astar"

  def search(self):
  
    # Early goal test for initial state
    p = self.search_problem

    min_heap = []
    heapq.heapify(min_heap)
    counter = itertools.count()  # Unique tie-breaker generator
    
    heapq.heappush(min_heap, (0 + self.w * p.h(p.initial_state), next(counter), SearchNode(p.initial_state, None, 0)))
    self.generated += 1

    reached = {}

    while min_heap:
        
        self.time_limit_reached()
        
        _, _, node = heapq.heappop(min_heap)  # Unpack (cost, tie-breaker, node)
        self.expanded += 1

        if node.state not in reached or node.g < reached[node.state]:
           
          reached[node.state] = node.g

          if p.is_goal(node.state):
            return self.extract_path(node), node.g         

          for action in p.actions(node.state):
              
              succ, cost = p.result(node.state, action)
              new_g = node.g + cost
              new_f = new_g + self.w * p.h(succ)
              succ_node = SearchNode(succ, node, new_g)

              heapq.heappush(min_heap, (new_f, next(counter), succ_node))
              self.generated += 1

    # No solution found
    return None, None


if __name__ == "__main__":
  problem = pancake_problem.generate_random_problem(5)
  problem = PancakeProblem((1, 5, 6, 2, 4, 3))
  problem.dump()
  astar = WeightedAStarSearch(problem, 1, print_statistics=True)
  astar.run()



