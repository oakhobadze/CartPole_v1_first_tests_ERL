import gymnasium as gym
import numpy as np
import time


class SimpleNNPolicy:
    def __init__(self, input_size, output_size, hidden_size=32, discrete=True):
        self.discrete = discrete
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, obs):
        obs = np.atleast_1d(obs).astype(np.float32)
        self.last_input = obs
        self.last_h = np.tanh(np.dot(obs, self.w1) + self.b1)
        output = np.dot(self.last_h, self.w2) + self.b2
        return output

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def act(self, obs):
        obs = np.atleast_1d(obs).astype(np.float32)
        output = self.forward(obs)
        if self.discrete:
            probs = self.softmax(output)
            return int(np.argmax(probs))
        else:
            return np.tanh(output)

    def act_stochastic(self, obs):
        obs = np.atleast_1d(obs).astype(np.float32)
        output = self.forward(obs)
        if self.discrete:
            probs = self.softmax(output)
            return np.random.choice(len(probs), p=probs)
        else:
            return np.tanh(output) + np.random.randn(*output.shape) * 0.1

    def mutate(self, mutation_scale=0.05):
        new_policy = SimpleNNPolicy(self.input_size, self.output_size, self.hidden_size, self.discrete)

        def mutate_param(param):
            return param + np.random.randn(*param.shape) * mutation_scale

        new_policy.w1 = mutate_param(self.w1)
        new_policy.b1 = mutate_param(self.b1)
        new_policy.w2 = mutate_param(self.w2)
        new_policy.b2 = mutate_param(self.b2)
        return new_policy

    @staticmethod
    def crossover(parent1, parent2):
        child = SimpleNNPolicy(parent1.input_size, parent1.output_size,
                               parent1.hidden_size, parent1.discrete)
        for attr in ['w1', 'b1', 'w2', 'b2']:
            p1_param = getattr(parent1, attr)
            p2_param = getattr(parent2, attr)
            mask = np.random.rand(*p1_param.shape) < 0.5
            setattr(child, attr, np.where(mask, p1_param, p2_param))
        return child


def evaluate_agent(env, agent, max_steps, render=False):
    total_reward = 0
    obs, _ = env.reset()
    terminated, truncated = False, False
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if render:
            env.render()
            time.sleep(0.02)
    return total_reward


def train_agent_with_rl(env, agent, episodes=5, learning_rate=0.001, max_steps=1000, gamma=0.99):
    discrete = hasattr(env.action_space, 'n')

    for _ in range(episodes):
        obs, _ = env.reset()
        states, actions, rewards, hiddens = [], [], [], []
        terminated, truncated = False, False
        steps = 0

        while not (terminated or truncated) and steps < max_steps:
            states.append(np.atleast_1d(obs).astype(np.float32).copy())
            if discrete:
                action = agent.act_stochastic(obs)
            else:
                action = agent.act(obs)
                action = action + np.random.randn(*action.shape) * 0.1
                action = np.clip(action, env.action_space.low, env.action_space.high)
            actions.append(action)
            hiddens.append(agent.last_h.copy())
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            steps += 1

        if len(rewards) == 0:
            continue

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        for state, action, G, h in zip(states, actions, returns, hiddens):
            output = np.dot(h, agent.w2) + agent.b2

            if discrete:
                probs = agent.softmax(output)
                d_output = -probs.copy()
                d_output[int(action)] += 1
                d_output *= G * learning_rate
            else:
                d_output = G * learning_rate * np.ones_like(output) * 0.01

            agent.w2 += np.outer(h, d_output)
            agent.b2 += d_output

            d_h = np.dot(agent.w2, d_output) * (1 - h ** 2)
            agent.w1 += np.outer(state, d_h)
            agent.b1 += d_h


def evolutionary_strategies_with_rl(env, population_size=50, generations=100, max_steps=1000,
                                     rl_episodes=5, hidden_size=32, mutation_scale=0.05,
                                     visualize=False, render_every=5):

    # Handle both vector and integer observation spaces
    if hasattr(env.observation_space, 'shape') and len(env.observation_space.shape) > 0:
        n_inputs = env.observation_space.shape[0]
    else:
        n_inputs = 1

    if hasattr(env.action_space, 'n'):
        n_outputs = env.action_space.n
        discrete = True
    else:
        n_outputs = env.action_space.shape[0]
        discrete = False

    population = [SimpleNNPolicy(n_inputs, n_outputs, hidden_size, discrete)
                  for _ in range(population_size)]
    avg_per_gen = []
    best_ever_agent = None
    best_ever_reward = -np.inf

    render_env = None
    if visualize:
        env_name = env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else env.spec.id
        render_env = gym.make(env_name, render_mode="human")

    for gen in range(generations):
        print(f"Generation {gen}: Training population with RL...")
        for agent in population:
            train_agent_with_rl(env, agent, episodes=rl_episodes,
                                learning_rate=0.001, max_steps=max_steps)

        rewards = [evaluate_agent(env, ind, max_steps=max_steps) for ind in population]

        best_idx = np.argmax(rewards)
        best_agent = population[best_idx]

        if rewards[best_idx] > best_ever_reward:
            best_ever_reward = rewards[best_idx]
            best_ever_agent = best_agent
            print(f"  *** New best ever: {best_ever_reward:.2f} ***")

        elite_idx = np.argsort(rewards)[-population_size // 5:]
        elites = [population[i] for i in elite_idx]

        new_population = elites.copy()
        new_population.append(best_ever_agent)

        current_mutation_scale = mutation_scale * (0.995 ** gen)
        current_mutation_scale = max(current_mutation_scale, 0.005)

        while len(new_population) < population_size:
            if np.random.rand() < 0.3:
                p1, p2 = np.random.choice(elites, 2, replace=False)
                child = SimpleNNPolicy.crossover(p1, p2)
            else:
                parent = np.random.choice(elites)
                child = parent.mutate(current_mutation_scale)
            new_population.append(child)

        avg_per_gen.append(np.mean(rewards))
        population = new_population

        print(f"  Best: {max(rewards):.2f}, Avg: {np.mean(rewards):.2f}, "
              f"Median: {np.median(rewards):.2f}, Mutation: {current_mutation_scale:.4f}, "
              f"Best Ever: {best_ever_reward:.2f}")

        if visualize and render_env is not None:
            if gen % render_every == 0 or gen == generations - 1:
                print(f"  --> Visualizing best agent from generation {gen}...")
                evaluate_agent(render_env, best_ever_agent, max_steps=max_steps, render=True)
                time.sleep(0.5)

    if visualize and render_env is not None:
        render_env.close()
    return avg_per_gen


if __name__ == "__main__":
    #env = gym.make("Pendulum-v1")
    env = gym.make("Acrobot-v1")
    #env = gym.make("LunarLander-v3")
    # env = gym.make("CartPole-v1")
    print(evolutionary_strategies_with_rl(env, population_size=100, generations=200,
                                          max_steps=1000, rl_episodes=10, hidden_size=64,
                                          mutation_scale=0.02, visualize=False, render_every=5))
    env.close()