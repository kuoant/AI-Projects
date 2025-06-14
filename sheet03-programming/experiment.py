from breadth_first_search import BreadthFirstSearch
from contextlib import contextmanager
from depth_first_search import DepthFirstSearch
from pancake_problem import PancakeProblem
from weighted_astar_search import WeightedAStarSearch

SIMPLE_PANCAKE_PROBLEMS = [
  (1, 5, 6, 2, 4, 3),
  (4, 1, 2, 5, 7, 3, 6),
  (6, 4, 3, 8, 7, 1, 2, 5),
  (7, 9, 8, 6, 1, 2, 5, 3, 4),
  (7, 8, 4, 1, 2, 9, 3, 6, 5, 10),
]

HARD_PANCAKE_PROBLEMS = [
  (10, 12, 7, 16, 4, 8, 1, 13, 18, 5, 15, 2, 20, 6, 17, 3, 9, 14, 11, 19),
  (14, 4, 3, 19, 13, 17, 16, 26, 2, 8, 28, 24, 9, 21, 12, 11, 25, 18, 5,
    15, 20, 23, 29, 10, 30, 7, 27, 22, 6, 1),
  (17, 29, 22, 16, 40, 20, 3, 10, 24, 36, 13, 30, 38, 34, 4, 26, 19, 7,
    21, 28, 25, 2, 8, 11, 35, 37, 15, 6, 27, 18, 31, 12, 33, 14, 23, 5,
    39, 1, 9, 32),
  (29, 38, 20, 32, 41, 43, 7, 39, 26, 13, 14, 1, 31, 5, 27, 35, 10, 21,
    2, 33, 6, 24, 23, 3, 45, 4, 16, 15, 30, 25, 11, 19, 12, 28, 8, 17,
    18, 9, 44, 34, 37, 36, 40, 42, 22),
  (32, 41, 44, 23, 33, 1, 28, 39, 10, 48, 34, 7, 21, 50, 27, 37, 30, 8,
    18, 31, 40, 5, 22, 17, 38, 43, 14, 47, 15, 3, 6, 4, 19, 49, 9, 16,
    35, 2, 20, 25, 36, 42, 13, 29, 45, 46, 24, 12, 11, 26),
]



def run_uninformed_searches(problem):
  print("\nDepth-first search:")
  dfs = DepthFirstSearch(problem, print_statistics=True)
  dfs.run()

  print("\nBreadth-first search:")
  bfs = BreadthFirstSearch(problem, print_statistics=True)
  bfs.run()


class TimeLimitExceededError(Exception):
  pass


def run_informed_searches(problem):
  for w in [1, 1.2, 1.5, 2]:
    print(f"\nWeighted A* Search with w={w}:")
    try:
      wastar = WeightedAStarSearch(problem, w, print_statistics=True,
                                   time_limit=180)
      wastar.run()
    except TimeLimitExceededError as e:
      print("Exceeded time limit of 3 minutes...")


for init in SIMPLE_PANCAKE_PROBLEMS:
  problem = PancakeProblem(init)
  problem.dump()

  run_uninformed_searches(problem)
  run_informed_searches(problem)
  print("----------")

for init in HARD_PANCAKE_PROBLEMS:
  problem = PancakeProblem(init)
  problem.dump()

  run_informed_searches(problem)
  print("----------")

