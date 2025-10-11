import gymnasium as gym
import numpy as np


class ActionSequence:
    def __init__(self, length):
        self.sequence = np.random.randint(0, 2, size=length)

    def mutate(self):
        new_seq = self.sequence.copy()
        num_mutations = np.random.randint(1, 4)
        indices = np.random.choice(len(new_seq), size=num_mutations, replace=False)
        new_seq[indices] = 1 - new_seq[indices]
        mutated = ActionSequence(len(new_seq))
        mutated.sequence = new_seq
        return mutated


def evaluate_sequence(env, action_seq):
    total_reward = 0
    obs, _ = env.reset()
    terminated, truncated = False, False
    idx = 0
    max_steps = len(action_seq.sequence)
    while not (terminated or truncated) and idx < max_steps:
        action = int(action_seq.sequence[idx])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        idx += 1
    return total_reward


def evolutionary_actions(env, population_size, generations, seq_length):
    population = [ActionSequence(seq_length) for _ in range(population_size)]
    avg_per_gen = []
    for gen in range(generations):
        rewards = [evaluate_sequence(env, ind) for ind in population]
        elite_idx = np.argsort(rewards)[-population_size // 5:]
        elites = [population[i] for i in elite_idx]
        new_population = elites.copy()
        while len(new_population) < population_size:
            parent = np.random.choice(elites)
            child = parent.mutate()
            new_population.append(child)
        population = new_population
        avg_per_gen.append(np.mean(rewards))
        print(
            f"Gen {gen}, Best: {max(rewards)}, "
            f"Avg: {np.mean(rewards):.2f}, Median: {np.median(rewards):.2f}"
        )
    return avg_per_gen


env = gym.make("CartPole-v1")
evolutionary_actions(env, population_size=30, generations=20, seq_length=250)
