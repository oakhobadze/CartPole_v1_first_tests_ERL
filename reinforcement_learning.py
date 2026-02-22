import gymnasium as gym
import numpy as np

def q_learning_main(episodes_per_generation=150, generations=20, alpha=0.2, gamma=0.99, epsilon_start=0.999):
    #env = gym.make("CartPole-v1")
    env = gym.make("LunarLander-v3")
    n_bins = [20, 30, 20, 30]
    bins = [np.linspace(-4.8, 4.8, n_bins[0]-1),
            np.linspace(-0.5, 0.5, n_bins[1]-1),
            np.linspace(-0.418, 0.418, n_bins[2]-1),
            np.linspace(-1, 1, n_bins[3]-1)]

    def discretize(obs):
        return tuple(np.digitize(obs[i], bins[i]) for i in range(4))

    q_table = np.zeros(n_bins + [env.action_space.n])
    epsilon = epsilon_start

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
                    action = np.argmax(q_table[state])
                obs, reward, terminated, truncated, _ = env.step(action)
                new_state = discretize(obs)
                q_table[state + (action,)] += alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state + (action,)])
                state = new_state
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
        epsilon = max(0.01, epsilon * 0.999)
        avg_per_gen.append(np.mean(rewards))
        print(f"Gen {gen}, Best: {max(rewards)}, Avg: {np.mean(rewards):.2f}, Median: {np.median(rewards):.2f}")
    env.close()
    return avg_per_gen

q_learning_main()