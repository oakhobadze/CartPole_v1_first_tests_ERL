import gymnasium as gym
import numpy as np
import time


class SimpleNNPolicy:
    def __init__(self, input_size, output_size, hidden_size=32, discrete=True):
        self.discrete = discrete
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros(output_size)

    def act(self, obs):
        obs = np.atleast_1d(obs).astype(np.float32)
        h = np.tanh(np.dot(obs, self.w1) + self.b1)
        output = np.dot(h, self.w2) + self.b2
        if self.discrete:
            return int(np.argmax(output))
        else:
            return np.tanh(output)

    def mutate(self, mutation_scale=0.05):
        new_policy = SimpleNNPolicy(self.w1.shape[0], self.w2.shape[1], self.w1.shape[1], self.discrete)

        def mutate_param(param):
            return param + np.random.randn(*param.shape) * mutation_scale

        new_policy.w1 = mutate_param(self.w1)
        new_policy.b1 = mutate_param(self.b1)
        new_policy.w2 = mutate_param(self.w2)
        new_policy.b2 = mutate_param(self.b2)
        return new_policy


def crossover(parent1, parent2):
    child = SimpleNNPolicy(parent1.w1.shape[0], parent1.w2.shape[1], parent1.w1.shape[1], parent1.discrete)
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


def evolutionary_strategies(env, population_size=100, generations=200, max_steps=1000,
                             hidden_size=32, mutation_scale=0.05, visualize=False, render_every=5):

    print(f"Running evolutionary_strategies on: {env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else env.spec.id}")

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

    population = [SimpleNNPolicy(n_inputs, n_outputs, hidden_size, discrete) for _ in range(population_size)]
    avg_per_gen = []
    best_ever_agent = None
    best_ever_reward = -np.inf

    render_env = None
    if visualize:
        env_name = env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else env.spec.id
        render_env = gym.make(env_name, render_mode="human")

    for gen in range(generations):
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
                child = crossover(p1, p2)
            else:
                parent = np.random.choice(elites)
                child = parent.mutate(current_mutation_scale)
            new_population.append(child)

        avg_per_gen.append(np.mean(rewards))
        population = new_population

        print(f"Generation {gen}, Best: {max(rewards):.2f}, Avg: {np.mean(rewards):.2f}, "
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
    #env = gym.make("MountainCar-v0")
    env = gym.make("Acrobot-v1")
    #env = gym.make("LunarLander-v3")
    #env = gym.make("CartPole-v1")
    print(evolutionary_strategies(env, population_size=200, generations=300, max_steps=500,
                                  hidden_size=64, mutation_scale=0.02, visualize=False, render_every=5))
    env.close()