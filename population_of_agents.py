import gymnasium as gym
import numpy as np

class SimpleNNPolicy:
    def __init__(self, input_size, hidden_size=8):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.w2 = np.random.randn(hidden_size)
        self.b2 = np.random.randn(1)

    def act(self, obs):
        h = np.tanh(np.dot(obs, self.w1) + self.b1)
        output = np.dot(h, self.w2) + self.b2
        return int(output > 0)

    def mutate(self):
        new_policy = SimpleNNPolicy(self.w1.shape[0], self.w1.shape[1])
        new_policy.w1 = self.w1 + 0.1 * np.random.randn(*self.w1.shape)
        new_policy.b1 = self.b1 + 0.1 * np.random.randn(*self.b1.shape)
        new_policy.w2 = self.w2 + 0.1 * np.random.randn(*self.w2.shape)
        new_policy.b2 = self.b2 + 0.1 * np.random.randn(*self.b2.shape)
        return new_policy

def evaluate_agent(env, agent, max_steps):
    total_reward = 0
    obs, _ = env.reset()
    terminated, truncated = False, False
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward

def evolutionary_strategies(env, population_size, generations, max_steps):
    n_inputs = env.observation_space.shape[0]
    population = [SimpleNNPolicy(n_inputs) for _ in range(population_size)]
    avg_per_gen = []
    for gen in range(generations):
        rewards = [evaluate_agent(env, ind, max_steps=max_steps) for ind in population]
        elite_idx = np.argsort(rewards)[-population_size//5:]
        elites = [population[i] for i in elite_idx]
        new_population = elites.copy()
        while len(new_population) < population_size:
            parent = np.random.choice(elites)
            child = parent.mutate()
            new_population.append(child)
        avg_per_gen.append(np.mean(rewards))
        population = new_population
        print(f"Generation {gen}, Best: {max(rewards)}, Avg: {np.mean(rewards):.2f}, Median: {np.median(rewards):.2f}")
    return avg_per_gen


env = gym.make("CartPole-v1")
print(evolutionary_strategies(env, population_size=30, generations=20, max_steps=250))
