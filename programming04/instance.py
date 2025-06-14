#! /usr/bin/env python3

class State:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id

    def __str__(self):
        return '(id:{}, x:{}, y:{})'.format(self.id, self.x, self.y)

    def __repr__(self):
        return '(id:{}, x:{}, y:{})'.format(self.id, self.x, self.y)

class Instance:
    def __init__(self, width, height, succ_probs, rewards):
        self.width = width
        self.height = height

        assert len(succ_probs) == self.height
        for succ_prob in succ_probs:
            assert len(succ_prob) == self.width

        assert len(rewards) == self.height
        for reward in rewards:
            assert len(reward) == self.width

        self.actions = ['N', 'E', 'S', 'W']

        self.states = [State(x,y,(y*self.width+x)) for y in range(self.height) for x in range(self.width)]
        self.init = self.states[0]
        self.goal = self.states[-1]

        self.succ_probs = { self.get_state(x,y) : succ_probs[y][x] for y in range(self.height) for x in range(self.width) }
        self.rewards =  { self.get_state(x,y) : rewards[y][x] for y in range(self.height) for x in range(self.width) }


    def get_state(self, x, y):
        assert x >= 0 and x < self.width and y >= 0 and y < self.height

        return self.states[y * self.width + x]


    def state_is_legal(self, state):
        return state.x >= 0 and state.x < self.width and state.y >= 0 and state.y < self.height

    def action_is_applicable(self, state, action):
        return (state != self.goal) and \
            ((action == 'N' and state.y < self.height-1) or \
             (action == 'E' and state.x < self.width-1) or \
             (action == 'S' and state.y > 0) or \
             (action == 'W' and state.x > 0))

    def get_applicable_actions(self, state):
        assert self.state_is_legal(state)
        return [action for action in self.actions if self.action_is_applicable(state, action)]

    def get_successors(self, state, action):
        assert self.state_is_legal(state)
        assert self.action_is_applicable(state, action)
        assert action in ['N', 'E', 'S', 'W']

        succ_prob = self.succ_probs[state]
        succs = []
        if succ_prob < 1.0:
            succs.append((state, 1.0 - succ_prob))

        if action == 'N':
            succs.append((self.get_state(state.x, state.y+1), succ_prob))
        elif action == 'E':
            succs.append((self.get_state(state.x+1, state.y), succ_prob))
        elif action == 'S':
            succs.append((self.get_state(state.x, state.y-1), succ_prob))
        else:
            succs.append((self.get_state(state.x-1, state.y), succ_prob))

        return succs

    # Careful: in y-direction, we print in reverse order s.t. coordinates
    # are as in a typical coordinate system
    def __str__(self):
        res = "Probabilities that moving is successful:\n"
        for y in reversed(range(self.height)):
            res += "[ " + "  ".join("{:.1f}".format(self.succ_probs[self.get_state(x,y)])
                for x in range(self.width)) + " ]\n"

        res += "\nCosts to move away from a cell:\n"
        for y in reversed(range(self.height)):
            res += "[ " + "  ".join("{:.0f}".format(self.rewards[self.get_state(x,y)])
                for x in range(self.width)) + " ]\n"

        return res


def get_example_instance():
    probs = [[1.0, 1.0, 1.0, 0.4],
             [0.4, 0.4, 1.0, 0.4],
             [0.4, 1.0, 1.0, 0.4],
             [0.4, 1.0, 0.4, 0.4],
             [0.4, 1.0, 1.0, 1.0]]

    rewards = [[-1.0, -1.0, -1.0, -1.0],
             [-1.0, -1.0, -1.0, -1.0],
             [-1.0, -1.0, -1.0, -1.0],
             [-1.0, -1.0, -3.0, -1.0],
             [-1.0, -1.0, -1.0, -1.0]]

    return Instance(4, 5, probs, rewards)


if __name__ == '__main__':
    inst = get_example_instance()

    print(inst)

    states = inst.states
    for s in states:
        print('applicable in {}: {}'.format(s, inst.get_applicable_actions(s)))

    print('')
    for s in states:
        for a in inst.get_applicable_actions(s):
            print('successors of applying {} in {} are {}'.format(a, s, inst.get_successors(s, a)))
        print('')

