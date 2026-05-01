import gymnasium as gym
import numpy as np
import time


def q_learning_main(env=None, episodes_per_generation=150, generations=50, alpha=0.1,
                    gamma=0.99, epsilon_start=1.0, n_bins_per_dim=10,
                    visualize=False, render_every=5):

    close_env = False
    if env is None:
        env = gym.make("LunarLander-v3")
        close_env = True

    print(f"Running q_learning on: {env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else env.spec.id}")
    # Handle both vector and integer observation spaces
    if hasattr(env.observation_space, 'shape') and len(env.observation_space.shape) > 0:
        obs_dim = env.observation_space.shape[0]
        obs_low = np.where(np.isinf(env.observation_space.low), -10, env.observation_space.low)
        obs_high = np.where(np.isinf(env.observation_space.high), 10, env.observation_space.high)
    else:
        obs_dim = 1
        obs_low = np.array([0])
        obs_high = np.array([env.observation_space.n - 1])

    n_actions = env.action_space.n
    n_bins = [n_bins_per_dim] * obs_dim
    bins = [np.linspace(obs_low[i], obs_high[i], n_bins[i] - 1) for i in range(obs_dim)]

    def discretize(obs):
        obs = np.atleast_1d(obs)
        return tuple(np.digitize(obs[i], bins[i]) for i in range(obs_dim))

    q_table = np.zeros(n_bins + [n_actions])
    epsilon = epsilon_start
    best_ever_reward = -np.inf

    render_env = None
    if visualize:
        env_name = env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else env.spec.id
        render_env = gym.make(env_name, render_mode="human")

    avg_per_gen = []

    for gen in range(generations):
        rewards = []

        for ep in range(episodes_per_generation):
            obs, _ = env.reset()
            state = discretize(obs)
            done = False
            total_reward = 0

            while not done:
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(q_table[state]))

                obs, reward, terminated, truncated, _ = env.step(action)
                new_state = discretize(obs)

                best_next = np.max(q_table[new_state])
                q_table[state + (action,)] += alpha * (
                    reward + gamma * best_next - q_table[state + (action,)]
                )

                state = new_state
                total_reward += reward
                done = terminated or truncated

            rewards.append(total_reward)

        epsilon = max(0.01, epsilon * 0.97)

        avg_reward = np.mean(rewards)
        best_gen_reward = max(rewards)
        avg_per_gen.append(avg_reward)

        if best_gen_reward > best_ever_reward:
            best_ever_reward = best_gen_reward
            print(f"  *** New best ever: {best_ever_reward:.2f} ***")

        print(f"Gen {gen}, Best: {best_gen_reward:.2f}, Avg: {avg_reward:.2f}, "
              f"Median: {np.median(rewards):.2f}, Epsilon: {epsilon:.4f}, "
              f"Best Ever: {best_ever_reward:.2f}")

        if visualize and render_env is not None:
            if gen % render_every == 0 or gen == generations - 1:
                print(f"  --> Visualizing best policy from generation {gen}...")
                obs, _ = render_env.reset()
                state = discretize(obs)
                done = False
                while not done:
                    action = int(np.argmax(q_table[state]))
                    obs, _, terminated, truncated, _ = render_env.step(action)
                    state = discretize(obs)
                    done = terminated or truncated
                    render_env.render()
                    time.sleep(0.02)
                time.sleep(0.5)

    if visualize and render_env is not None:
        render_env.close()

    if close_env:
        env.close()

    return avg_per_gen


if __name__ == "__main__":
    #env = gym.make("Pendulum-v1")
    # env = gym.make("LunarLander-v3")
    # env = gym.make("CartPole-v1")
    env = gym.make("Acrobot-v1")
    # env = gym.make("MountainCar-v0")
    print(q_learning_main(env=env, episodes_per_generation=300, generations=100,
                          alpha=0.2, gamma=0.99, epsilon_start=1.0, n_bins_per_dim=15,
                          visualize=False, render_every=5))
    env.close()