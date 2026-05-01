import gymnasium as gym
import numpy as np
import time
from collections import deque
import random


class SimpleNN:
    def __init__(self, in_size, out_size, lr=0.001):
        self.w1 = np.random.randn(in_size, 128) * np.sqrt(2.0 / in_size)
        self.b1 = np.zeros(128)
        self.w2 = np.random.randn(128, 128) * np.sqrt(2.0 / 128)
        self.b2 = np.zeros(128)
        self.w3 = np.random.randn(128, out_size) * np.sqrt(2.0 / 128)
        self.b3 = np.zeros(out_size)
        self.lr = lr

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.w1 + self.b1
        self.h1 = np.maximum(0, self.z1)
        self.z2 = self.h1 @ self.w2 + self.b2
        self.h2 = np.maximum(0, self.z2)
        self.out = self.h2 @ self.w3 + self.b3
        return self.out

    def backward(self, x, target, output):
        batch_size = len(x)

        dout = (output - target) / batch_size
        dw3 = self.h2.T @ dout
        db3 = np.sum(dout, axis=0)

        dh2 = dout @ self.w3.T
        dz2 = dh2 * (self.z2 > 0)
        dw2 = self.h1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        dh1 = dz2 @ self.w2.T
        dz1 = dh1 * (self.z1 > 0)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.w3 -= self.lr * dw3
        self.b3 -= self.lr * db3
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

    def copy_from(self, other):
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()
        self.w3 = other.w3.copy()
        self.b3 = other.b3.copy()


class DQNAgent:
    def __init__(self, state_dim, action_dim, alpha=0.0005, gamma=0.99, epsilon=1.0,
                 buffer_size=100000, batch_size=64):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.batch_size = batch_size
        self.q_net = SimpleNN(state_dim, action_dim, alpha)
        self.target_net = SimpleNN(state_dim, action_dim, alpha)
        self.target_net.copy_from(self.q_net)
        self.buffer = deque(maxlen=buffer_size)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        return int(np.argmax(self.q_net.forward(state.reshape(1, -1))))

    def store(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def train(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        s = np.array([b[0] for b in batch])
        a = np.array([b[1] for b in batch])
        r = np.array([b[2] for b in batch])
        s_next = np.array([b[3] for b in batch])
        done = np.array([b[4] for b in batch])

        q_current = self.q_net.forward(s)
        q_next = np.max(self.target_net.forward(s_next), axis=1)
        q_target = q_current.copy()
        q_target[np.arange(self.batch_size), a] = r + (1 - done) * self.gamma * q_next

        self.q_net.backward(s, q_target, q_current)

        return np.mean((q_current[np.arange(self.batch_size), a] -
                        q_target[np.arange(self.batch_size), a]) ** 2)

    def update_target(self):
        self.target_net.copy_from(self.q_net)


def evaluate_agent(env, agent, max_steps, render=False):
    obs, _ = env.reset()
    obs = np.atleast_1d(obs).astype(np.float32)
    total_reward = 0
    for _ in range(max_steps):
        action = int(np.argmax(agent.q_net.forward(obs.reshape(1, -1))))
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = np.atleast_1d(obs).astype(np.float32)
        total_reward += reward
        if render:
            env.render()
            time.sleep(0.02)
        if terminated or truncated:
            break
    return total_reward


def dqn_algorithm(env, alpha=0.0005, gamma=0.99, epsilon=1.0, buffer_size=100000,
                  batch_size=64, target_update_steps=500, generations=100,
                  episodes_per_gen=10, max_steps=1000, visualize=False, render_every=10):

    # Handle both vector and integer observation spaces
    if hasattr(env.observation_space, 'shape') and len(env.observation_space.shape) > 0:
        state_dim = env.observation_space.shape[0]
    else:
        state_dim = 1

    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, alpha, gamma, epsilon, buffer_size, batch_size)
    avg_per_gen = []
    best_ever_reward = -np.inf
    total_steps = 0

    render_env = None
    if visualize:
        env_name = env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else env.spec.id
        render_env = gym.make(env_name, render_mode="human")

    print("Filling replay buffer...")
    warmup_obs, _ = env.reset()
    warmup_obs = np.atleast_1d(warmup_obs).astype(np.float32)
    for _ in range(min(5000, buffer_size // 10)):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.atleast_1d(next_obs).astype(np.float32)
        agent.store(warmup_obs, action, reward, next_obs, float(terminated or truncated))
        warmup_obs = next_obs
        if terminated or truncated:
            warmup_obs, _ = env.reset()
            warmup_obs = np.atleast_1d(warmup_obs).astype(np.float32)
    print(f"Buffer prefilled with {len(agent.buffer)} experiences.")

    for gen in range(generations):
        gen_rewards = []

        for _ in range(episodes_per_gen):
            obs, _ = env.reset()
            obs = np.atleast_1d(obs).astype(np.float32)
            ep_reward = 0

            for _ in range(max_steps):
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs = np.atleast_1d(next_obs).astype(np.float32)
                done = terminated or truncated

                agent.store(obs, action, reward, next_obs, float(done))
                agent.train()

                obs = next_obs
                ep_reward += reward
                total_steps += 1

                if total_steps % target_update_steps == 0:
                    agent.update_target()

                if done:
                    break

            gen_rewards.append(ep_reward)

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.97)

        avg_reward = np.mean(gen_rewards)
        best_gen_reward = max(gen_rewards)
        avg_per_gen.append(avg_reward)

        if best_gen_reward > best_ever_reward:
            best_ever_reward = best_gen_reward
            print(f"  *** New best ever: {best_ever_reward:.2f} ***")

        print(f"Generation {gen}, Best: {best_gen_reward:.2f}, Avg: {avg_reward:.2f}, "
              f"Median: {np.median(gen_rewards):.2f}, Epsilon: {agent.epsilon:.4f}, "
              f"Buffer: {len(agent.buffer)}, Steps: {total_steps}, "
              f"Best Ever: {best_ever_reward:.2f}")

        if visualize and render_env is not None:
            if gen % render_every == 0 or gen == generations - 1:
                print(f"  --> Visualizing best agent from generation {gen}...")
                evaluate_agent(render_env, agent, max_steps, render=True)
                time.sleep(0.5)

    if visualize and render_env is not None:
        render_env.close()

    return avg_per_gen


if __name__ == "__main__":
    #env = gym.make("Taxi-v3")
    env = gym.make("LunarLander-v3")
    # env = gym.make("CartPole-v1")
    #env = gym.make("Acrobot-v1")
    # env = gym.make("MountainCar-v0")
    print(dqn_algorithm(env, alpha=0.0003, gamma=0.99, epsilon=1.0, buffer_size=200000,
                        batch_size=128, target_update_steps=1000, generations=200,
                        episodes_per_gen=20, max_steps=500, visualize=False, render_every=10))
    env.close()