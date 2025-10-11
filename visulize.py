import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from population_of_agents import evolutionary_strategies
from population_of_actions import evolutionary_actions
from reinforcement_learning import q_learning_main

runs = 10
generations = 20
env = gym.make("CartPole-v1")
steps = 250
alpha=0.2
gamma=0.99
epsilon_start=0.999

def repeat_and_average(run_func, **kwargs):
    all_runs = [run_func(**kwargs) for _ in range(runs)]
    return np.mean(all_runs, axis=0)

agents_avg = repeat_and_average(evolutionary_strategies, env=env,population_size=20, generations=generations, max_steps=steps)
actions_avg = repeat_and_average(evolutionary_actions, env=env, population_size=20, generations=generations, seq_length=steps)
qlearning_avg = repeat_and_average(q_learning_main, episodes_per_generation=150, generations=generations, alpha=alpha,gamma=gamma,epsilon_start=epsilon_start)

x = np.arange(1, generations + 1)

plt.plot(x, agents_avg, label="Population of Agents")
plt.plot(x, actions_avg, label="Population of Actions")
plt.plot(x, qlearning_avg, label="Q-learning")
plt.xlabel("Generation")
plt.ylabel("Average resultgit ")
plt.title("CartPole-v1: Algorithm Comparison")
plt.legend()
plt.grid(True)
plt.show()
