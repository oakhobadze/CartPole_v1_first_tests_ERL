import gymnasium as gym
import numpy as np


def discretize_obs(obs, bins, low, high):
    obs = np.clip(obs, low, high)
    grid = [np.linspace(l, h, n - 1) for l, h, n in zip(low, high, bins)]
    return tuple(int(np.digitize(o, g)) for o, g in zip(obs, grid))


class TabularPolicy:
    def __init__(self, state_shape, n_actions=2):
        self.policy_table = np.random.randint(0, n_actions, size=state_shape)
        self.n_actions = n_actions

    def act(self, state):
        return self.policy_table[state]

    def mutate(self, mutation_rate=0.05):
        new_policy = TabularPolicy(self.policy_table.shape, self.n_actions)
        new_policy.policy_table = self.policy_table.copy()
        n_mutations = int(np.prod(self.policy_table.shape) * mutation_rate)
        for _ in range(n_mutations):
            idx = tuple(np.random.randint(0, s) for s in self.policy_table.shape)
            new_policy.policy_table[idx] = np.random.randint(0, self.n_actions)
        return new_policy

    @staticmethod
    def crossover(parent1, parent2):
        child = TabularPolicy(parent1.policy_table.shape, parent1.n_actions)
        mask = np.random.rand(*parent1.policy_table.shape) < 0.5
        child.policy_table = np.where(mask, parent1.policy_table, parent2.policy_table)
        return child


def evaluate_policy(env, policy, bins, low, high, max_steps=500):
    total_reward = 0
    obs, _ = env.reset()
    terminated, truncated = False, False
    steps = 0
    while not (terminated or truncated) and steps < max_steps:
        state = discretize_obs(obs, bins, low, high)
        action = policy.act(state)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward


def evolutionary_policies(env, population_size=20, generations=20):
    bins = (6, 12, 6, 12)
    low = np.array([-4.8, -4, -0.418, -4])
    high = np.array([4.8, 4, 0.418, 4])

    state_shape = tuple(b for b in bins)
    population = [TabularPolicy(state_shape) for _ in range(population_size)]
    avg_per_gen = []

    for gen in range(generations):
        rewards = [evaluate_policy(env, ind, bins, low, high) for ind in population]

        elite_idx = np.argsort(rewards)[-population_size // 5:]
        elites = [population[i] for i in elite_idx]

        new_population = elites.copy()
        while len(new_population) < population_size:
            if np.random.rand() < 0.5:
                p1, p2 = np.random.choice(elites, 2, replace=False)
                child = TabularPolicy.crossover(p1, p2)
            else:
                parent = np.random.choice(elites)
                child = parent.mutate()
            new_population.append(child)

        population = new_population
        avg_reward = np.mean(rewards)
        avg_per_gen.append(avg_reward)
        print(f"Gen {gen}, Best: {max(rewards):.2f}, Avg: {avg_reward:.2f}")

    return avg_per_gen


env = gym.make("CartPole-v1")
evolutionary_policies(env, population_size=20, generations=20)
