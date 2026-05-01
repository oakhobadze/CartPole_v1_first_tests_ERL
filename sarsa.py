import gymnasium as gym
import numpy as np
import time


class SARSAAgent:
    def __init__(self, actions, obs_dim, obs_low, obs_high, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.obs_dim = obs_dim
        self.obs_low = obs_low
        self.obs_high = obs_high
        self.q_table = {}

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def get_all_q(self, state):
        return np.array([self.get_q(state, a) for a in self.actions])

    def discretize(self, obs, n_bins=10):
        obs = np.atleast_1d(obs)
        state = []
        for i, val in enumerate(obs):
            clipped = np.clip(val, self.obs_low[i], self.obs_high[i])
            bin_idx = min(
                int(np.digitize(clipped, np.linspace(self.obs_low[i], self.obs_high[i], n_bins - 1))),
                n_bins - 1
            )
            state.append(bin_idx)
        return tuple(state)

    def get_action(self, state):
        if np.random.rand() > self.epsilon:
            return int(np.argmax(self.get_all_q(state)))
        return np.random.choice(self.actions)

    def update(self, s, a, r, s_next, a_next, done):
        q_pred = self.get_q(s, a)
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * self.get_q(s_next, a_next)
        self.q_table[(s, a)] = q_pred + self.alpha * (q_target - q_pred)


def evaluate_agent(env, agent, max_steps, n_bins=10, render=False):
    total_reward = 0
    obs, _ = env.reset()
    state = agent.discretize(obs, n_bins)
    terminated, truncated = False, False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        action = int(np.argmax(agent.get_all_q(state)))
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = agent.discretize(obs, n_bins)
        total_reward += reward
        state = next_state
        steps += 1
        if render:
            env.render()
            time.sleep(0.02)

    return total_reward


def sarsa_algorithm(env, n_bins=10, alpha=0.1, gamma=0.99, epsilon=1.0,
                    generations=50, episodes_per_gen=100, max_steps=1000,
                    visualize=False, render_every=5):

    # Handle both vector and integer observation spaces
    if hasattr(env.observation_space, 'shape') and len(env.observation_space.shape) > 0:
        obs_dim = env.observation_space.shape[0]
        obs_low = np.where(np.isinf(env.observation_space.low), -10, env.observation_space.low)
        obs_high = np.where(np.isinf(env.observation_space.high), 10, env.observation_space.high)
    else:
        # Discrete observation space like Taxi, FrozenLake
        obs_dim = 1
        obs_low = np.array([0])
        obs_high = np.array([env.observation_space.n - 1])

    agent = SARSAAgent(
        actions=list(range(env.action_space.n)),
        obs_dim=obs_dim,
        obs_low=obs_low,
        obs_high=obs_high,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )
    avg_per_gen = []
    best_ever_reward = -np.inf

    render_env = None
    if visualize:
        env_name = env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else env.spec.id
        render_env = gym.make(env_name, render_mode="human")

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

        agent.epsilon = max(0.01, agent.epsilon * 0.97)

        avg_reward = np.mean(gen_rewards)
        best_gen_reward = max(gen_rewards)
        avg_per_gen.append(avg_reward)

        if best_gen_reward > best_ever_reward:
            best_ever_reward = best_gen_reward
            print(f"  *** New best ever: {best_ever_reward:.2f} ***")

        print(f"Generation {gen}, Best: {best_gen_reward:.2f}, Avg: {avg_reward:.2f}, "
              f"Median: {np.median(gen_rewards):.2f}, Epsilon: {agent.epsilon:.4f}, "
              f"Best Ever: {best_ever_reward:.2f}, Q-table size: {len(agent.q_table)}")

        if visualize and render_env is not None:
            if gen % render_every == 0 or gen == generations - 1:
                print(f"  --> Visualizing best agent from generation {gen}...")
                evaluate_agent(render_env, agent, max_steps=max_steps, n_bins=n_bins, render=True)
                time.sleep(0.5)

    if visualize and render_env is not None:
        render_env.close()

    return avg_per_gen


if __name__ == "__main__":
    #env = gym.make("Pendulum-v1")
    #env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v3")
    env = gym.make("Acrobot-v1")
    # env = gym.make("MountainCar-v0")
    print(sarsa_algorithm(env, n_bins=15, alpha=0.2, gamma=0.99, epsilon=1.0,
                          generations=100, episodes_per_gen=250, max_steps=1000,
                          visualize=False, render_every=5))
    env.close()