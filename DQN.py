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

        # Output layer
        dout = (output - target) / batch_size
        dw3 = self.h2.T @ dout
        db3 = np.sum(dout, axis=0)

        # Hidden layer 2
        dh2 = dout @ self.w3.T
        dz2 = dh2 * (self.z2 > 0)
        dw2 = self.h1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # Hidden layer 1
        dh1 = dz2 @ self.w2.T
        dz1 = dh1 * (self.z1 > 0)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Update weights
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
                 buffer_size=50000, batch_size=32):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = batch_size
        self.q_net = SimpleNN(state_dim, action_dim, alpha)
        self.target_net = SimpleNN(state_dim, action_dim, alpha)
        self.target_net.copy_from(self.q_net)
        self.buffer = deque(maxlen=buffer_size)

    def get_action(self, state):
        if np.random.rand() > self.epsilon:  # Explore with probability (1-epsilon)
            return np.random.randint(self.action_dim)
        return np.argmax(self.q_net.forward(state.reshape(1, -1)))

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

        # Current Q values
        q_current = self.q_net.forward(s)

        # Target Q values
        q_next = np.max(self.target_net.forward(s_next), axis=1)
        q_target = q_current.copy()
        q_target[np.arange(self.batch_size), a] = r + (1 - done) * self.gamma * q_next

        # Train
        self.q_net.backward(s, q_target, q_current)

        return np.mean((q_current[np.arange(self.batch_size), a] -
                        q_target[np.arange(self.batch_size), a]) ** 2)

    def update_target(self):
        self.target_net.copy_from(self.q_net)


def evaluate_agent(env, agent, max_steps, render=False):
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(max_steps):
        action = np.argmax(agent.q_net.forward(obs.reshape(1, -1)))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if render:
            env.render()
            time.sleep(0.02)
        if terminated or truncated:
            break
    return total_reward


def dqn_algorithm(env, alpha=0.0005, gamma=0.99, epsilon=1.0, buffer_size=50000,
                  batch_size=32, target_update=100, generations=100, episodes_per_gen=5,
                  max_steps=500, visualize=False, render_every=10):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, alpha, gamma, epsilon, buffer_size, batch_size)
    avg_per_gen = []

    render_env = None
    if visualize:
        try:
            render_env = gym.make("CartPole-v1", render_mode="human")
        except:
            visualize = False

    total_steps = 0

    for gen in range(generations):
        gen_rewards = []

        for _ in range(episodes_per_gen):
            obs, _ = env.reset()
            ep_reward = 0

            for _ in range(max_steps):
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store(obs, action, reward, next_obs, float(done))

                if len(agent.buffer) >= batch_size:
                    agent.train()

                obs = next_obs
                ep_reward += reward
                total_steps += 1

                if done:
                    break

            gen_rewards.append(ep_reward)

        # Update target network every few generations
        if gen % 5 == 0:
            agent.update_target()

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        avg_per_gen.append(np.mean(gen_rewards))

        print(f"Generation {gen}, Best: {max(gen_rewards):.2f}, Avg: {np.mean(gen_rewards):.2f}, "
              f"Median: {np.median(gen_rewards):.2f}, Epsilon: {agent.epsilon:.4f}, Buffer: {len(agent.buffer)}")

        if visualize and render_env and (gen % render_every == 0 or gen == generations - 1):
            print(f"  --> Visualizing best agent from generation {gen}...")
            evaluate_agent(render_env, agent, max_steps, render=True)
            time.sleep(0.5)

    if render_env:
        render_env.close()

    return avg_per_gen

env = gym.make("LunarLander-v3")
#env = gym.make("CartPole-v1")
print(dqn_algorithm(env, alpha=0.0005, gamma=0.99, epsilon=1.0, buffer_size=50000,
                    batch_size=32, target_update=100, generations=40, episodes_per_gen=5,
                    max_steps=500, visualize=False, render_every=10))
env.close()