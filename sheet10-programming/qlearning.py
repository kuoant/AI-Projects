import numpy as np
from tqdm import tqdm
from utils import get_epsilon_greedy_action, get_legal_actions
from rl_agent import RLAgent

class QLearning(RLAgent):
    def __init__(self, rng, initialize_q_values, name, n_episodes, alpha, gamma, epsilon, epsilon_decay=False) -> None:
        super().__init__(rng, initialize_q_values, name, n_episodes, alpha, gamma, epsilon)
        self.epsilon_decay = epsilon_decay

    def update_q_value(self, s, a, s_prime, r, alpha, gamma):
        """Q-Learning update rule (off-policy)."""
        legal_actions = get_legal_actions(s_prime)
        max_q_next = np.max(self.q_values[s_prime, legal_actions])
        self.q_values[s, a] += alpha * (r + gamma * max_q_next - self.q_values[s, a])

    def train(self, env):
        print(f"Training {self.name}...")
        progress_bar = tqdm(np.arange(self.n_episodes + 1))
        for episode in progress_bar:
            if episode == self.n_episodes:
                progress_bar.set_description(f'Simulating greedy policy')
                epsilon = 0
            else:
                if self.epsilon_decay:
                    epsilon = self.epsilon * (1 - episode / self.n_episodes)
                else:
                    epsilon = self.epsilon

            s, _ = env.reset()
            is_terminal = False
            episode_reward = 0

            while not is_terminal:
                a = get_epsilon_greedy_action(env.np_random, s, self.q_values, epsilon)
                s_prime, r, terminated, truncated, _ = env.step(a)
                is_terminal = terminated or truncated
                self.update_q_value(s, a, s_prime, r, self.alpha, self.gamma)

                s = s_prime
                episode_reward += r

            self.total_reward_per_episode[episode] = episode_reward
            self.render(env, episode, progress_bar, episode_reward)

        print(f"Training {self.name} done.")
