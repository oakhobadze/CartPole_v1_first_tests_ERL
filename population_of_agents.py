import gymnasium as gym
import numpy as np
import time


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

    def mutate(self, mutation_scale=0.1):
        new_policy = SimpleNNPolicy(self.w1.shape[0], self.w1.shape[1])

        def mutate_param(param):
            return param * np.random.uniform(1 - mutation_scale, 1 + mutation_scale, size=param.shape)

        new_policy.w1 = mutate_param(self.w1)
        new_policy.b1 = mutate_param(self.b1)
        new_policy.w2 = mutate_param(self.w2)
        new_policy.b2 = mutate_param(self.b2)
        return new_policy


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
            time.sleep(0.02)  # Slow down for visibility
    return total_reward


def evolutionary_strategies(env, population_size, generations, max_steps, visualize=False, render_every=5):
    n_inputs = env.observation_space.shape[0]
    population = [SimpleNNPolicy(n_inputs) for _ in range(population_size)]
    avg_per_gen = []

    # Create rendering environment only if visualization is enabled
    render_env = None
    if visualize:
        render_env = gym.make("CartPole-v1", render_mode="human")

    for gen in range(generations):
        rewards = [evaluate_agent(env, ind, max_steps=max_steps) for ind in population]

        # Find best agent
        best_idx = np.argmax(rewards)
        best_agent = population[best_idx]

        # Evolution step
        elite_idx = np.argsort(rewards)[-population_size // 5:]
        elites = [population[i] for i in elite_idx]
        new_population = elites.copy()
        while len(new_population) < population_size:
            parent = np.random.choice(elites)
            child = parent.mutate()
            new_population.append(child)
        avg_per_gen.append(np.mean(rewards))
        population = new_population

        print(
            f"Generation {gen}, Best: {max(rewards):.2f}, Avg: {np.mean(rewards):.2f}, Median: {np.median(rewards):.2f}")
        if visualize and render_env is not None:
            if gen % render_every == 0 or gen == generations - 1:
                print(f"  --> Visualizing best agent from generation {gen}...")
                evaluate_agent(render_env, best_agent, max_steps=max_steps, render=True)
                time.sleep(0.5)  # Pause between generations

    if visualize and render_env is not None:
        render_env.close()
    return avg_per_gen



env = gym.make("CartPole-v1")
print(evolutionary_strategies(env, population_size=20, generations=40, max_steps=500, visualize=False, render_every=5))

env.close()