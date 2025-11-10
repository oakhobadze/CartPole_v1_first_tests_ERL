import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from population_of_agents import evolutionary_strategies
from population_of_actions import evolutionary_actions
from reinforcement_learning import q_learning_main
from population_of_policies import evolutionary_policies

runs = 10
generations = 40
env = gym.make("CartPole-v1")
steps = 500
alpha = 0.2
gamma = 0.99
epsilon_start = 0.999
max_r = 500


def repeat_and_average(run_func, **kwargs):
    all_runs = [run_func(**kwargs) for _ in range(runs)]
    return np.mean(all_runs, axis=0)


agents_avg = repeat_and_average(
    evolutionary_strategies,
    env=env,
    population_size=20,
    generations=generations,
    max_steps=steps,
)

actions_avg = repeat_and_average(
    evolutionary_actions,
    env=env,
    population_size=20,
    generations=generations,
    max_seq_length=max_r,
)

qlearning_avg = repeat_and_average(
    q_learning_main,
    episodes_per_generation=150,
    generations=generations,
    alpha=alpha,
    gamma=gamma,
    epsilon_start=epsilon_start,
)

policies_avg = repeat_and_average(
    evolutionary_policies,
    env=env,
    population_size=20,
    generations=generations,
)


x = np.arange(1, generations + 1)

plt.figure(figsize=(10, 6))
plt.plot(x, agents_avg, label="Population of Agents")
plt.plot(x, actions_avg, label="Population of Actions")
plt.plot(x, qlearning_avg, label="Q-learning")
plt.plot(x, policies_avg, label="Population of Policies", linestyle="--", linewidth=2)

plt.xlabel("Generation")
plt.ylabel("Average reward")
plt.title("CartPole-v1: Comparison of Evolutionary and RL Algorithms")
plt.legend()
plt.grid(True)
plt.show()
