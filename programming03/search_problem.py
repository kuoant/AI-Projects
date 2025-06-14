import random


class SearchProblem:
  """
  This class provides a simple interface for search problems. States are
  numbers (=IDs) from 0 to *states - 1*. The initial state is given as
  *initial_state* and *goal_states* is a list of goal states. The
  transitions come as an adjacency list *transitions* with *states*
  entries, one for every state. Each entry is again a list of tuples
  *(succ_id, cost)*, one for every successor.
  """
  def __init__(self, states, init, goals, transitions):
    assert(states == len(transitions))
    assert(0 <= init < states)
    assert(0 <= goal < states for goal in goals)
    assert((0 <= succ < states for succ, _ in transitions[s]) for s in
      range(states))
    self.states = states
    self.initial_state = init
    self.goal_states = goals
    self.transitions = transitions

  def initial_state(self):
    return self.initial_state

  def is_goal(self, state):
    return state in self.goal_states

  def actions(self, state):
    return [(state, succ) for succ in self.transitions[state].keys()]

  def result(self, state, action):
    assert(len(action) == 2)
    assert(state == action[0])
    succ = action[1]
    cost = self.transitions[state][succ]
    return succ, cost

  def _get_dot_string(self):
    digraph = "digraph G {\n  init [shape=point];\n"

    for state in range(self.states):
      digraph += f"  s{state} [label=\"s{state}\""
      if state in self.goal_states:
        digraph += ", style=\"bold,filled\""
      digraph += "];\n"

      if state == self.initial_state:
        digraph += f"    init -> s{state};\n"

      for action in self.actions(state):
        succ, cost = self.result(state, action)
        digraph += f"    s{state} -> s{succ} [label=\"{cost}\"];\n"

    digraph += "}\n"

    return digraph


  def dump(self, filename=None):
    digraph = self._get_dot_string()
    if filename:
      with open(filename, "w") as f:
        f.write(digraph)
      """
      To visualize the search problem, you can execute "dot" (if
      installed on your system) as follows:

          dot -Tpng <filename> -o digraph.png

      and then open the PNG.
      """
    else:
      print(digraph)



def generate_random_problem(states, min_successors, max_successors,
                            goal_states=1, max_cost=1):
  assert(states >= goal_states > 0)
  assert(min_successors <= max_successors)

  def generate_random_goals():
    return random.sample(range(states), goal_states)

  def generate_random_successors(state):
    num_successors = random.randint(min_successors, max_successors)

    successors = {}
    for succ_id in random.sample(range(states), num_successors):
      cost = random.randint(1, max_cost)
      successors[succ_id] = cost

    return successors

  def generate_random_transitions():
    transitions = []
    for state in range(states):
      transitions.append(generate_random_successors(state))
    return transitions

  goals = generate_random_goals()
  transitions = generate_random_transitions()

  return SearchProblem(states, 0, goals, transitions)


def __main__():
  rand = generate_random_problem(3, 1, 2)
  rand.dump()


if __name__== "__main__":
  __main__()

