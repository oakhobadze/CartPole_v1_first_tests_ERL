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

def repeat_and_collect(run_func, **kwargs):
    """Run algorithm multiple times and collect all rewards."""
    all_runs = [run_func(**kwargs) for _ in range(runs)]
    return np.array(all_runs)  # shape: (runs, generations)

# Збір результатів
agents_all = repeat_and_collect(
    evolutionary_strategies,
    env=env,
    population_size=20,
    generations=generations,
    max_steps=steps,
)

actions_all = repeat_and_collect(
    evolutionary_actions,
    env=env,
    population_size=20,
    generations=generations,
    max_seq_length=max_r,
)

qlearning_all = repeat_and_collect(
    q_learning_main,
    episodes_per_generation=150,
    generations=generations,
    alpha=alpha,
    gamma=gamma,
    epsilon_start=epsilon_start,
)

policies_all = repeat_and_collect(
    evolutionary_policies,
    env=env,
    population_size=20,
    generations=generations,
)

# Обчислення середнього та std dev
def compute_mean_std(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return mean, std

agents_mean, agents_std = compute_mean_std(agents_all)
actions_mean, actions_std = compute_mean_std(actions_all)
qlearning_mean, qlearning_std = compute_mean_std(qlearning_all)
policies_mean, policies_std = compute_mean_std(policies_all)

x = np.arange(1, generations + 1)

# Побудова графіку з std dev
plt.figure(figsize=(12, 7))
plt.plot(x, agents_mean, label="Population of Agents")co
plt.fill_between(x, agents_mean - agents_std, agents_mean + agents_std, alpha=0.2)

plt.plot(x, actions_mean, label="Population of Actions")
plt.fill_between(x, actions_mean - actions_std, actions_mean + actions_std, alpha=0.2)

plt.plot(x, qlearning_mean, label="Q-learning")
plt.fill_between(x, qlearning_mean - qlearning_std, qlearning_mean + qlearning_std, alpha=0.2)

plt.plot(x, policies_mean, label="Population of Policies", linestyle="--", linewidth=2)
plt.fill_between(x, policies_mean - policies_std, policies_mean + policies_std, alpha=0.2)

plt.xlabel("Generation")
plt.ylabel("Average reward")
plt.title("CartPole-v1: Comparison of Evolutionary and RL Algorithms (with Std Dev)")
plt.legend()
plt.grid(True)
plt.show()

# Процентний ріст кожної 5-ї генерації
def compute_percent_growth(data_mean, step=5):
    growth = []
    prev = 0
    for gen in range(step-1, len(data_mean), step):
        if prev == 0:
            pct = (data_mean[gen] - 0) * 100 / max_r  # Для першої точки порівнюємо з 0
        else:
            pct = (data_mean[gen] - data_mean[prev-1]) / data_mean[prev-1] * 100
        growth.append((gen+1, pct))  # (номер генерації, процентний ріст)
        prev = gen + 1
    return growth

# Приклад для всіх алгоритмів
agents_growth = compute_percent_growth(agents_mean)
actions_growth = compute_percent_growth(actions_mean)
qlearning_growth = compute_percent_growth(qlearning_mean)
policies_growth = compute_percent_growth(policies_mean)

# Виведення результатів
print("Percentage growth every 5 generations:")
print("Population of Agents:", agents_growth)
print("Population of Actions:", actions_growth)
print("Q-learning:", qlearning_growth)
print("Population of Policies:", policies_growth)
