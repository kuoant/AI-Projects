#%%
import gymnasium as gym
import os
from sarsa import SARSA
from qlearning import QLearning
from utils import plot_reward_functions



if __name__ == '__main__':

    # environment parameters
    n_episodes = 300    # number of episodes to train each agent
    alpha = 0.1         # learning rate
    gamma = 0.9         # discount rate
    epsilon = 0.25      # probability of exploration

    # initialize environment
    env = gym.make('CliffWalking-v0', render_mode='rgb_array_list')
    env.reset(seed=21)

    # make directory to store renderings of the training process
    if not os.path.exists('Results'):
        os.mkdir('Results')
    
    sarsa = SARSA(
        name='SARSA',
        rng=env.np_random,
        initialize_q_values='optimistic',
        n_episodes=n_episodes, 
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon,
        epsilon_decay=False
    )
    sarsa.train(env)

    sarsa_epsilon_decay = SARSA(
        name='SARSA (epsilon decay)',
        rng=env.np_random,
        initialize_q_values='optimistic',
        n_episodes=n_episodes, 
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon,
        epsilon_decay=True
    )
    sarsa_epsilon_decay.train(env)

    qlearning = QLearning(
        name="Q-Learning",
        rng=env.np_random,
        initialize_q_values='optimistic',
        n_episodes=n_episodes, 
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon
    )
    qlearning.train(env)

    # plot reward functions
    plot_reward_functions(n_episodes, [sarsa, qlearning], 'fixed epsilon')
    plot_reward_functions(n_episodes, [sarsa, sarsa_epsilon_decay], 'epsilon decay')

    env.close()
# %%
