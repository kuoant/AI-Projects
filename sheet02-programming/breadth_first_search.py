import search_problem

from queue import Queue
from search import Search, SearchNode


class BreadthFirstSearch(Search):
  name = "breadth-first"

  def search(self):
    # early goal test for initial state
    p = self.search_problem
    if p.is_goal(p.initial_state):
      return [p.initial_state], 0

    # enqueue initial state
    frontier = Queue()
    frontier.put(SearchNode(p.initial_state, None, 0))
    self.generated += 1
    reached = {p.initial_state}

    while not frontier.empty():
      node = frontier.get()
      self.expanded += 1

      for action in p.actions(node.state):
        succ, cost = p.result(node.state, action)
        new_g = node.g + cost
        succ_node = SearchNode(succ, node, new_g)

        # early goal test
        if p.is_goal(succ):
          return self.extract_path(succ_node), new_g

        # mark reached to avoid cycles
        if succ not in reached:
          reached.add(succ)

          # enqueue successor
          frontier.put(succ_node)
          self.generated += 1

    # no solution found
    return None, None


if __name__ == "__main__":
  problem = search_problem.generate_random_problem(8, 2, 3)
  problem.dump()
  bfs = BreadthFirstSearch(problem, True)
  bfs.run()


