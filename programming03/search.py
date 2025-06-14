import search_problem
import time


class SearchNode:
  def __init__(self, state, parent, g):
    self.state = state
    self.parent = parent
    self.g = g


class Search:
  def __init__(self, search_problem, print_statistics=False,
               time_limit=None):
    self.search_problem = search_problem
    self.expanded = 0
    self.generated = 0
    self.print_statistics = print_statistics
    self.max_generations = 1000000
    self.time_limit = time_limit
    self.begin_time = None

  def search(self):
    raise NotImplementedError

  def time_limit_reached(self):
    if self.time_limit is None:
      return False

    return time.time() - self.begin_time > self.time_limit

  def run(self):
    self.expanded = 0
    self.generated = 0

    self.begin_time = time.time()
    print(f"Starting {self.name} search...")

    plan, cost = self.search()
    if plan is not None:
      print(f"Found solution of cost {cost}.")

    total_time = time.time() - self.begin_time
    print(f"Search time: {total_time:.3f}")

    if self.print_statistics:
      print(f"Expanded states: {self.expanded}")
      print(f"Generated states: {self.generated}")
      #if plan is not None:
      # print(f"State sequence: {plan}")

  def extract_path(self, node):
    path = [node.state]
    while node.parent:
      node = node.parent
      path.append(node.state)

    path.reverse()
    return path
