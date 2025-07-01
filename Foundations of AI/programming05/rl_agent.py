import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class RLAgent:
    def __init__(self, rng, initialize_q_values, name, n_episodes, alpha, gamma, epsilon) -> None:
        """
        Parameters:
            alpha (float):  learning rate, should be in the interval [0,1]
            gamma (float):  discount rate, should be in the interval [0,1]
        """
        self.name = name
        # table of Q-values (often also known as Q-Table), dim: (states, actions)
        if initialize_q_values == 'random':
            # initialize values randomly between [-100, 100)
            self.q_values = rng.random((48, 4))*200 - 100
        elif initialize_q_values == 'optimistic':
            self.q_values = np.zeros((48, 4)) 

        # hyperparameters
        self.n_episodes = n_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # some stored data
        self.total_reward_per_episode = np.full(n_episodes+1, np.nan)
        self.snapshot_episodes = np.array([1, 5, 10, 25, 100, 300])-1  # 0-based indices


    def render_episode(self, array_list, name, episode_reward):
        fig, ax = plt.subplots()

        frames = [[ax.imshow(im, animated=True)] for im in array_list]
        ax.text(x=0.005, y=1.15, s=name+f' (total reward: {episode_reward})', transform=ax.transAxes, fontsize=12, fontfamily='serif', va='center')

        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=1000)
        ani.save('Results/'+name+'.mp4')
        plt.close(fig)


    def render(self, env, episode, progress_bar, episode_reward):
        if episode in self.snapshot_episodes:
            progress_bar.set_description(f'Episode: {episode+1}, rendering and saving video...')
            array_list = env.render()
            self.render_episode(array_list, f'{self.name} Episode {episode+1}', episode_reward)

        # for the appended extra episode store the render
        if episode == self.n_episodes:
            progress_bar.set_description(f'Simulating greedy policy, rendering and saving video...')
            array_list = env.render()
            self.render_episode(array_list, f'{self.name} Greedy Policy', episode_reward)
