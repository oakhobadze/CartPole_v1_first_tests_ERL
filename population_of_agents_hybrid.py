import gymnasium as gym
import numpy as np


class SimpleNNPolicy:
    def __init__(self, input_size, hidden_size=8):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.w2 = np.random.randn(hidden_size)
        self.b2 = np.random.randn(1)

    def act(self, obs, return_logits=False):
        h = np.tanh(np.dot(obs, self.w1) + self.b1)
        output = np.dot(h, self.w2) + self.b2
        if return_logits:
            return output
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

    def copy(self):
        """Create a copy of the policy"""
        new_policy = SimpleNNPolicy(self.w1.shape[0], self.w1.shape[1])
        new_policy.w1 = self.w1.copy()
        new_policy.b1 = self.b1.copy()
        new_policy.w2 = self.w2.copy()
        new_policy.b2 = self.b2.copy()
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


def reinforce_episode(env, agent, max_steps, learning_rate=0.001):
    """
    Run one episode and collect trajectory for REINFORCE update
    Returns: total_reward, updated_agent
    """
    observations = []
    actions = []
    rewards = []

    obs, _ = env.reset()
    terminated, truncated = False, False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        logit = agent.act(obs, return_logits=True)
        prob_action_1 = 1 / (1 + np.exp(-logit))
        action = 1 if np.random.random() < prob_action_1 else 0

        observations.append(obs)
        actions.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        steps += 1

    # Calculate returns (simple monte carlo)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = np.array(returns)

    # Normalize returns
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # REINFORCE update
    updated_agent = agent.copy()
    for obs, action, G in zip(observations, actions, returns):
        # Forward pass
        h = np.tanh(np.dot(obs, updated_agent.w1) + updated_agent.b1)
        logit = np.dot(h, updated_agent.w2) + updated_agent.b2
        prob_action_1 = 1 / (1 + np.exp(-logit))

        # Calculate gradient of log probability
        if action == 1:
            grad_logit = (1 - prob_action_1)
        else:
            grad_logit = -prob_action_1

        # Backprop
        grad_w2 = h * grad_logit * G * learning_rate
        grad_b2 = grad_logit * G * learning_rate

        dh = updated_agent.w2 * grad_logit * G * learning_rate
        dh_tanh = dh * (1 - h ** 2)

        grad_w1 = np.outer(obs, dh_tanh) * learning_rate
        grad_b1 = dh_tanh * learning_rate

        # Update weights
        updated_agent.w1 += grad_w1
        updated_agent.b1 += grad_b1
        updated_agent.w2 += grad_w2
        updated_agent.b2 += grad_b2

    return sum(rewards), updated_agent


def hybrid_evolutionary_rl(env, population_size, generations, max_steps, rl_episodes=3, learning_rate=0.001):
    """
    Hybrid approach: Each generation, agents learn via RL before being evaluated for selection
    """
    n_inputs = env.observation_space.shape[0]
    population = [SimpleNNPolicy(n_inputs) for _ in range(population_size)]
    avg_per_gen = []

    for gen in range(generations):
        # RL LEARNING PHASE: Each agent learns via REINFORCE
        print(f"\nGeneration {gen} - RL Learning Phase...")
        for i, agent in enumerate(population):
            for ep in range(rl_episodes):
                _, updated_agent = reinforce_episode(env, agent, max_steps, learning_rate)
                population[i] = updated_agent

        # EVALUATION PHASE
        rewards = [evaluate_agent(env, ind, max_steps=max_steps) for ind in population]

        # SELECTION & REPRODUCTION PHASE
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

    return avg_per_gen


# Run the hybrid approach
env = gym.make("CartPole-v1")
results = hybrid_evolutionary_rl(
    env,
    population_size=20,
    generations=40,
    max_steps=500,
    rl_episodes=10,
    learning_rate=0.001
)