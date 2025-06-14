import random
import search_problem

from breadth_first_search import BreadthFirstSearch
from depth_first_search import DepthFirstSearch
from uniform_cost_search import UniformCostSearch

random.seed(2022)

print("Unit cost search:")
for i in range(50000, 100001, 10000):
  print(f"Generating search problem with {i} states...")
  problem = search_problem.generate_random_problem(i, 3, 5, max_cost=1)
  problem.dump(f"digraphs-{i}.dot")

  print("\nBreadth-first search:")
  bfs = BreadthFirstSearch(problem, print_statistics=True)
  bfs.run()

  print("\nUniform-cost search:")
  ucs = UniformCostSearch(problem, print_statistics=True)
  ucs.run()

  print("\nDepth-first search:")
  dfs = DepthFirstSearch(problem)
  dfs.run()

  print("----------")

