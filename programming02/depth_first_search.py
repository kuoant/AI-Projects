import search_problem

from search import Search, SearchNode


class DepthFirstSearch(Search):
  name = "depth-first"

  def search(self):
    # early goal test for initial state
    p = self.search_problem
    if p.is_goal(p.initial_state):
      return [p.initial_state], 0

    # enqueue initial state
    frontier = [SearchNode(p.initial_state, None, 0)]
    self.generated += 1
    reached = {p.initial_state}

    while frontier:
      node = frontier.pop()
      self.expanded += 1

      # mark reached to avoid cycles
      reached.add(node.state)

      for action in p.actions(node.state):
        succ, cost = p.result(node.state, action)
        if succ in reached:
          continue

        new_g = node.g + cost
        succ_node = SearchNode(succ, node, new_g)

        # early goal test
        if p.is_goal(succ):
          return self.extract_path(succ_node), new_g

        # enqueue successor
        frontier.append(succ_node)
        self.generated += 1

    # no solution found
    return None, None


if __name__ == "__main__":
  problem = search_problem.generate_random_problem(8, 2, 3)
  problem.dump()
  dfs = DepthFirstSearch(problem, True)
  dfs.run()


