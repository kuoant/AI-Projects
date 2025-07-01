import random

from search_problem import SearchProblem


class PancakeProblem(SearchProblem):
  """
  This class defines a special domain of search problems. We consider a
  stack of pancakes, each of which has a different diameter (enumerated
  from 1 to n if there are n pancakes). Initially, the stack of pancakes
  is ordered arbitrarily and the goal is to sort the stack by size with
  the smallest pancake on top.
  """
  def __init__(self, init):
    self.pancakes = len(init)
    self.initial_state = init

  def initial_state(self):
    return self.initial_state

  def is_goal(self, state):
    """
    The (only) goal state of the pancake problem is the one where the
    positions, from top to bottom, are occupied by pancakes of
    increasing size: (1, 2, ..., n-1, n)
    """
    assert(len(state) == self.pancakes)
    for i in range(self.pancakes):
      if state[i] != i + 1:
        return False
    return True

  def actions(self, state):
    """
    Every action is defined by the number of pancakes to be swapped.
    """
    assert(len(state) == self.pancakes)
    return list(range(1, self.pancakes + 1))

  def result(self, state, action):
    assert(len(state) == self.pancakes)
    assert(0 <= action <= self.pancakes)

    succ = list(state)
    for i in range(action):
      succ[i] = state[action - i - 1]
    return tuple(succ), 1

  def h(self, state):
    """
    The idea of this heuristic is to count the pairs of adjacent
    pancakes that don't match. Since it requires at least one action to
    fix that flaw and no action fixes more than one flaw, this is an
    admissible heuristic.
    """
    count = 0 if state[-1] == self.pancakes else 1
    for i in range(self.pancakes - 1):
      diff = state[i] - state[i + 1]
      if diff > 1 or diff < -1:
        count += 1
    return count

  def _get_dot_string(self):
    raise NotImplementedError

  def dump(self, filename=None):
    print("Initial state:", self.initial_state)



def generate_random_problem(pancakes):
  init = list(range(1, pancakes + 1))
  random.shuffle(init)
  return PancakeProblem(tuple(init))


def __main__():
  rand = generate_random_problem(10)
  rand.dump()

if __name__== "__main__":
  __main__()

