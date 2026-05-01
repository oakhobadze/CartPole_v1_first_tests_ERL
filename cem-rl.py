import gymnasium as gym
import numpy as np
import time
from collections import deque
import random


# ── SimpleNN ──────────────────────────────────────────────────────────────────

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

    def get_params(self):
        return np.concatenate([
            self.w1.flatten(), self.b1.flatten(),
            self.w2.flatten(), self.b2.flatten(),
            self.w3.flatten(), self.b3.flatten()
        ])

    def set_params(self, params):
        idx = 0
        def take(shape):
            nonlocal idx
            size = int(np.prod(shape))
            arr = params[idx:idx + size].reshape(shape)
            idx += size
            return arr
        self.w1 = take(self.w1.shape)
        self.b1 = take(self.b1.shape)
        self.w2 = take(self.w2.shape)
        self.b2 = take(self.b2.shape)
        self.w3 = take(self.w3.shape)
        self.b3 = take(self.b3.shape)


# ── DQNAgent ──────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self, state_dim, action_dim, alpha=0.0005, gamma=0.99,
                 epsilon=1.0, buffer_size=100000, batch_size=64):
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
            return
        batch = random.sample(self.buffer, self.batch_size)
        s = np.array([b[0] for b in batch])
        a = np.array([b[1] for b in batch])
        r = np.array([b[2] for b in batch])
        s_next = np.array([b[3] for b in batch])
        done = np.array([b[4] for b in batch])
        q_current = self.q_net.forward(s)
        q_next = np.max(self.target_net.forward(s_next), axis=1)
        q_target = q_current.copy()
        q_target[np.arange(self.batch_size), a] = (
            r + (1 - done) * self.gamma * q_next
        )
        self.q_net.backward(s, q_target, q_current)

    def update_target(self):
        self.target_net.copy_from(self.q_net)


# ── CEM distribution ──────────────────────────────────────────────────────────

class CEMDistribution:
    def __init__(self, param_dim, std_init=0.5):
        self.param_dim = param_dim
        self.mean = np.zeros(param_dim)
        self.std = np.ones(param_dim) * std_init
        self._std_init = std_init  # keep for std floor reset

    def sample(self):
        return self.mean + self.std * np.random.randn(self.param_dim)

    def update(self, elite_params, std_min=0.01):
        elite_params = np.array(elite_params)
        self.mean = np.mean(elite_params, axis=0)
        self.std = np.std(elite_params, axis=0)
        self.std = np.maximum(self.std, std_min)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_network(env, network, max_steps=1000):
    obs, _ = env.reset()
    obs = np.atleast_1d(obs).astype(np.float32)
    total_reward = 0
    for _ in range(max_steps):
        q_values = network.forward(obs.reshape(1, -1))
        action = int(np.argmax(q_values))
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = np.atleast_1d(obs).astype(np.float32)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


def train_dqn_agent(env, agent, steps=1000, target_update_steps=500):
    obs, _ = env.reset()
    obs = np.atleast_1d(obs).astype(np.float32)
    total_steps = 0

    while total_steps < steps:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.atleast_1d(next_obs).astype(np.float32)
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, float(done))
        agent.train()

        obs = next_obs
        total_steps += 1

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.9995)

        if total_steps % target_update_steps == 0:
            agent.update_target()

        if done:
            obs, _ = env.reset()
            obs = np.atleast_1d(obs).astype(np.float32)

    # removed the per-generation epsilon decay that was here before


# ── Main CEM-RL algorithm ─────────────────────────────────────────────────────

def cem_rl(env, population_size=10, generations=100, max_steps=1000,
           dqn_steps_per_gen=1000, elite_fraction=0.5,
           alpha=0.0005, gamma=0.99, buffer_size=100000,
           batch_size=64, target_update_steps=500,
           std_init=0.5, std_min=0.01,
           eval_episodes=1,
           visualize=False, render_every=10):

    if hasattr(env.observation_space, 'shape') and len(env.observation_space.shape) > 0:
        state_dim = env.observation_space.shape[0]
    else:
        state_dim = 1

    action_dim = env.action_space.n
    n_elites = max(1, int(population_size * elite_fraction))

    print(f"Running CEM-RL on: {env.unwrapped.spec.id}")
    print(f"Population: {population_size}, Elites: {n_elites}, "
          f"DQN steps/gen: {dqn_steps_per_gen}")

    # ── Initialize DQN agent ──────────────────────────────────────────────────
    dqn_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        buffer_size=buffer_size,
        batch_size=batch_size
    )

    # Warmup replay buffer
    print("Filling replay buffer...")
    warmup_obs, _ = env.reset()
    warmup_obs = np.atleast_1d(warmup_obs).astype(np.float32)
    for _ in range(min(5000, buffer_size // 10)):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = np.atleast_1d(next_obs).astype(np.float32)
        dqn_agent.store(
            warmup_obs, action, reward,
            next_obs, float(terminated or truncated)
        )
        warmup_obs = next_obs
        if terminated or truncated:
            warmup_obs, _ = env.reset()
            warmup_obs = np.atleast_1d(warmup_obs).astype(np.float32)
    print(f"Buffer prefilled with {len(dqn_agent.buffer)} experiences.")

    # ── Initialize CEM distribution ───────────────────────────────────────────
    param_dim = len(dqn_agent.q_net.get_params())
    cem = CEMDistribution(param_dim=param_dim, std_init=std_init)
    cem.mean = dqn_agent.q_net.get_params().copy()

    avg_per_gen = []
    best_ever_reward = -np.inf
    best_ever_params = None

    render_env = None
    if visualize:
        env_name = (env.unwrapped.spec.id
                    if hasattr(env.unwrapped, 'spec') else env.spec.id)
        render_env = gym.make(env_name, render_mode="human")

    for gen in range(generations):

        # ── Step 1: Build population ──────────────────────────────────────────
        train_dqn_agent(
            env, dqn_agent,
            steps=dqn_steps_per_gen,
            target_update_steps=target_update_steps
        )

        n_cem = population_size // 2
        n_dqn = population_size - n_cem

        population = []

        # CEM-sampled agents
        for _ in range(n_cem):
            agent_net = SimpleNN(state_dim, action_dim)
            params = cem.sample()
            agent_net.set_params(params)
            population.append(agent_net)

        for _ in range(n_dqn):
            agent_net = SimpleNN(state_dim, action_dim)
            agent_net.copy_from(dqn_agent.q_net)
            params = agent_net.get_params()
            # noise scaled to current CEM spread so diversity stays meaningful
            noise_scale = np.mean(cem.std) * 0.3
            params += np.random.randn(len(params)) * noise_scale
            agent_net.set_params(params)
            population.append(agent_net)

        # ── Step 2: Evaluate all agents ───────────────────────────────────────
        rewards = []
        for agent_net in population:
            ep_rewards = [
                evaluate_network(env, agent_net, max_steps=max_steps)
                for _ in range(eval_episodes)
            ]
            r = np.mean(ep_rewards)
            rewards.append(r)

            obs, _ = env.reset()
            obs = np.atleast_1d(obs).astype(np.float32)
            for _ in range(min(500, max_steps)):
                q_values = agent_net.forward(obs.reshape(1, -1))
                action = int(np.argmax(q_values))
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs = np.atleast_1d(next_obs).astype(np.float32)
                dqn_agent.store(
                    obs, action, reward,
                    next_obs, float(terminated or truncated)
                )
                obs = next_obs
                if terminated or truncated:
                    break

        rewards = np.array(rewards)

        # ── Step 3: Update CEM distribution from elites ───────────────────────
        elite_idx = np.argsort(rewards)[-n_elites:]
        elite_params = [population[i].get_params() for i in elite_idx]
        cem.update(elite_params, std_min=std_min)

        # ── Step 4: Track best ever ───────────────────────────────────────────
        best_idx = np.argmax(rewards)
        best_gen_reward = rewards[best_idx]
        avg_reward = np.mean(rewards)
        median_reward = np.median(rewards)
        avg_per_gen.append(avg_reward)

        if best_gen_reward > best_ever_reward:
            best_ever_reward = best_gen_reward
            best_ever_params = population[best_idx].get_params().copy()
            print(f"  *** New best ever: {best_ever_reward:.2f} ***")

        print(f"Generation {gen}, Best: {best_gen_reward:.2f}, "
              f"Avg: {avg_reward:.2f}, Median: {median_reward:.2f}, "
              f"DQN epsilon: {dqn_agent.epsilon:.4f}, "
              f"CEM std mean: {np.mean(cem.std):.4f}, "
              f"Best Ever: {best_ever_reward:.2f}")

        if visualize and render_env is not None:
            if gen % render_every == 0 or gen == generations - 1:
                print(f"  --> Visualizing best agent from generation {gen}...")
                best_net = SimpleNN(state_dim, action_dim)
                best_net.set_params(best_ever_params)
                evaluate_network(render_env, best_net, max_steps=max_steps)
                time.sleep(0.5)

    if visualize and render_env is not None:
        render_env.close()

    return avg_per_gen


if __name__ == "__main__":
    for run in range(1):
        print(f"\nRun {run+1}/5")
        env = gym.make("LunarLander-v3")
        print(cem_rl(
            env,
            population_size=100,
            generations=200,
            max_steps=1000,
            dqn_steps_per_gen=5000,
            elite_fraction=0.3,
            alpha=0.0003,
            gamma=0.99,
            buffer_size=200000,
            batch_size=64,
            target_update_steps=500,
            std_init=0.5,
            std_min=0.05,
            eval_episodes=3,
            visualize=False,
            render_every=10
        ))
        env.close()