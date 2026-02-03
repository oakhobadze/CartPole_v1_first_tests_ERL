import gymnasium as gym
import numpy as np
import pandas as pd
import time


class SARSAAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon=0.9):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def check_state(self, state):
        if state not in self.q_table.index:
            self.q_table = pd.concat([self.q_table, pd.DataFrame([[0.0] * len(self.actions)],
                                                                 index=[state], columns=self.actions)])

    def discretize(self, obs, n_bins=10):
        bounds = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.21, 0.21], [-2.0, 2.0]])
        state = []
        for i, val in enumerate(obs):
            clipped = np.clip(val, bounds[i][0], bounds[i][1])
            bin_idx = min(int(np.digitize(clipped, np.linspace(bounds[i][0], bounds[i][1], n_bins - 1))), n_bins - 1)
            state.append(bin_idx)
        return str(tuple(state))

    def get_action(self, state):
        self.check_state(state)
        if np.random.rand() < self.epsilon:
            actions = self.q_table.loc[state, :].reindex(np.random.permutation(self.q_table.columns))
            return actions.idxmax()
        return np.random.choice(self.actions)

    def update(self, s, a, r, s_next, a_next, done):
        self.check_state(s_next)
        q_pred = self.q_table.loc[s, a]
        q_target = r if done else r + self.gamma * self.q_table.loc[s_next, a_next]
        self.q_table.loc[s, a] += self.alpha * (q_target - q_pred)


def evaluate_agent(env, agent, max_steps, n_bins=10, render=False):
    total_reward = 0
    obs, _ = env.reset()
    state = agent.discretize(obs, n_bins)
    terminated, truncated = False, False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        agent.check_state(state)
        action = agent.q_table.loc[state, :].idxmax()
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = agent.discretize(obs, n_bins)
        total_reward += reward
        state = next_state
        steps += 1
        if render:
            env.render()
            time.sleep(0.02)

    return total_reward


def sarsa_algorithm(env, n_bins=10, alpha=0.1, gamma=0.99, epsilon=0.9,
                    generations=40, episodes_per_gen=10, max_steps=500,
                    visualize=False, render_every=5):
    agent = SARSAAgent(list(range(env.action_space.n)), alpha, gamma, epsilon)
    avg_per_gen = []

    # Create rendering environment only if visualization is enabled
    render_env = None
    if visualize:
        render_env = gym.make("CartPole-v1", render_mode="human")

    for gen in range(generations):
        gen_rewards = []

        for _ in range(episodes_per_gen):
            obs, _ = env.reset()
            s = agent.discretize(obs, n_bins)
            a = agent.get_action(s)
            ep_reward = 0

            for _ in range(max_steps):
                obs, r, terminated, truncated, _ = env.step(a)
                s_next = agent.discretize(obs, n_bins)
                a_next = agent.get_action(s_next)
                agent.update(s, a, r, s_next, a_next, terminated or truncated)
                s, a = s_next, a_next
                ep_reward += r
                if terminated or truncated:
                    break

            gen_rewards.append(ep_reward)

        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        avg_per_gen.append(np.mean(gen_rewards))

        print(f"Generation {gen}, Best: {max(gen_rewards):.2f}, Avg: {np.mean(gen_rewards):.2f}, "
              f"Median: {np.median(gen_rewards):.2f}, Epsilon: {agent.epsilon:.4f}")

        if visualize and render_env is not None:
            if gen % render_every == 0 or gen == generations - 1:
                print(f"  --> Visualizing best agent from generation {gen}...")
                evaluate_agent(render_env, agent, max_steps=max_steps, n_bins=n_bins, render=True)
                time.sleep(0.5)

    if visualize and render_env is not None:
        render_env.close()

    return avg_per_gen


env = gym.make("CartPole-v1")
print(sarsa_algorithm(env, n_bins=10, alpha=0.1, gamma=0.99, epsilon=0.9,
                      generations=40, episodes_per_gen=10, max_steps=500,
                      visualize=True, render_every=5))
env.close()