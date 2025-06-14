import numpy as np
import matplotlib.pyplot as plt


def get_legal_actions(s):
    """Return list of legal actions for a state s."""
    legal_actions = []
    if s >= 12:  # if not on first row, allow moving up (0)
        legal_actions.append(0)
    if (s+1) % 12 != 0:  # if not on last column, allow moving right (1)
        legal_actions.append(1)
    if s < 36:  # if not on bottom row, allow moving down (2)
        legal_actions.append(2)
    if s % 12 != 0:  # if not on first column, allow moving left (3)
        legal_actions.append(3)
    return legal_actions


def get_epsilon_greedy_action(rng, s, q_values, epsilon):
    """
    Return an action chosen via epsilon greedy policy. Notice that in this version of epsilon greedy we could also by chance pick the greedy action.

    Arguments:
        rng (numpy random number generator): You should always pass env.np_random
        s (int): current state
        q_values (np.ndarray): array of state-action values (Q-values), dim=(48,4)
        epsilon (float): epsilon value, should be in the interval [0,1]
    """
    assert epsilon <= 1 and epsilon >= 0

    legal_actions = get_legal_actions(s)

    if rng.random() <= 1 - epsilon:  # take greedy action 
        action = legal_actions[np.argmax(q_values[s, legal_actions])]
    else:  # take random action
        action = rng.choice(legal_actions)

    return action


def exponential_moving_average(array, alpha=0.1):
    """
    Calculate exponential moving average (ema) of an array
    """
    ema = np.full(len(array), np.nan)
    ema[0] = array[0]
    for i in np.arange(1, len(array)):
        ema[i] = alpha * array[i] + (1-alpha) * ema[i-1]
    return ema


def plot_reward_functions(n_episodes, agents, name):
    """
    Parameters:
        n_episodes (int): number of training episodes
        agents (list(RLAgent)): list of trained agents
    """

    fig, ax = plt.subplots()

    x_axis = np.arange(n_episodes)
    for agent in agents:
        rewards = agent.total_reward_per_episode[:-1]
        ax.plot(x_axis, rewards, label=agent.name, alpha=0.7)
        ax.plot(x_axis, exponential_moving_average(rewards), label=agent.name+' (EMA)', alpha=1)

    # annotate plots
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.set_ylim(-800, 0)
    ax.set_title(f'Training Reward per Episode ({name})')
    ax.legend()
    fig.savefig(f'Results/Reward Functions ({name}).png')
    return None

