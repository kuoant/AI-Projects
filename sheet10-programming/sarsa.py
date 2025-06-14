import numpy as np
from tqdm import tqdm
from utils import get_epsilon_greedy_action
from rl_agent import RLAgent

class SARSA(RLAgent):
    def __init__(self, rng, initialize_q_values, name, n_episodes, alpha, gamma, epsilon, epsilon_decay=False) -> None:
        super().__init__(rng, initialize_q_values, name, n_episodes, alpha, gamma, epsilon)
        self.epsilon_decay = epsilon_decay

    def update_q_value(self, s, a, s_prime, a_prime, r, alpha, gamma):
        """SARSA Q-value update rule."""
        self.q_values[s, a] += alpha * (r + gamma * self.q_values[s_prime, a_prime] - self.q_values[s, a])

    def train(self, env):
        print(f"Training {self.name}...")
        progress_bar = tqdm(np.arange(self.n_episodes + 1))
        for episode in progress_bar:
            if episode == self.n_episodes:
                progress_bar.set_description(f'Simulating greedy policy')
                epsilon = 0  # act greedily
            else:
                if self.epsilon_decay:
                    epsilon = self.epsilon * (1 - episode / self.n_episodes)
                else:
                    epsilon = self.epsilon

            s, _ = env.reset()
            a = get_epsilon_greedy_action(env.np_random, s, self.q_values, epsilon)
            is_terminal = False
            episode_reward = 0

            while not is_terminal:
                s_prime, r, terminated, truncated, _ = env.step(a)
                is_terminal = terminated or truncated
                a_prime = get_epsilon_greedy_action(env.np_random, s_prime, self.q_values, epsilon)
                self.update_q_value(s, a, s_prime, a_prime, r, self.alpha, self.gamma)
                
                s = s_prime
                a = a_prime
                
                episode_reward += r

            self.total_reward_per_episode[episode] = episode_reward
            self.render(env, episode, progress_bar, episode_reward)

        print(f"Training {self.name} done.")
 