import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import json

from population_of_agents import evolutionary_strategies
from population_of_actions import evolutionary_actions
from reinforcement_learning import q_learning_main
from population_of_policies import evolutionary_policies
from population_of_agents_hybrid import evolutionary_strategies_with_rl
from sarsa import sarsa_algorithm
from DQN import dqn_algorithm
from cem_rl import cem_rl  # <-- add this import

runs = 5
#ENV_ID = "Taxi-v3"
#ENV_ID = "Pendulum-v1"
#ENV_ID = "MountainCar-v0"
#ENV_ID = "LunarLander-v3"
#ENV_ID = "Acrobot-v1"
ENV_ID = "CartPole-v1"

gens = 100

class MountainCarShapedEnv(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = reward + 300 * abs(obs[1])
        return obs, shaped_reward, terminated, truncated, info

def get_next_filename(base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(base_dir, f"test{i}.json")):
        i += 1
    return os.path.join(base_dir, f"test{i}.json")


def repeat_and_collect(run_func, env_id, **kwargs):
    all_runs = []
    for _ in range(runs):
        env = gym.make(env_id)
        result = run_func(env=env, **kwargs)
        env.close()
        all_runs.append(result)
    return np.array(all_runs)


# ── Neural Network Evolutionary ───────────────────────────────────────────────
agents_all = repeat_and_collect(
    evolutionary_strategies,
    env_id=ENV_ID,
    population_size=100,
    generations=gens,
    max_steps=1000,
    hidden_size=32,
    mutation_scale=0.05,
)
agents_gens = gens

# ── Hybrid (Evolutionary + RL) ────────────────────────────────────────────────
hybrid_all = repeat_and_collect(
    evolutionary_strategies_with_rl,
    env_id=ENV_ID,
    population_size=50,
    generations=gens,
    max_steps=1000,
    rl_episodes=5,
    hidden_size=32,
    mutation_scale=0.05,
)
hybrid_gens = gens

# ── Action Sequences ──────────────────────────────────────────────────────────
actions_all = repeat_and_collect(
    evolutionary_actions,
    env_id=ENV_ID,
    population_size=100,
    generations=gens,
    max_seq_length=500,
)
actions_gens = gens

# ── Q-Learning ────────────────────────────────────────────────────────────────
qlearning_all = repeat_and_collect(
    q_learning_main,
    env_id=ENV_ID,
    episodes_per_generation=250,
    generations=gens,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    n_bins_per_dim=10,
)
qlearning_gens = gens

# ── Tabular Policies ──────────────────────────────────────────────────────────
policies_all = repeat_and_collect(
    evolutionary_policies,
    env_id=ENV_ID,
    population_size=100,
    generations=gens,
    mutation_rate=0.05,
)
policies_gens = gens

# ── SARSA ─────────────────────────────────────────────────────────────────────
sarsa_all = repeat_and_collect(
    sarsa_algorithm,
    env_id=ENV_ID,
    n_bins=10,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    generations=gens,
    episodes_per_gen=100,
    max_steps=1000,
)
sarsa_gens = gens

# ── DQN ───────────────────────────────────────────────────────────────────────
dqn_all = repeat_and_collect(
    dqn_algorithm,
    env_id=ENV_ID,
    alpha=0.0005,
    gamma=0.99,
    epsilon=1.0,
    buffer_size=100000,
    batch_size=64,
    target_update_steps=500,
    generations=gens,
    episodes_per_gen=10,
    max_steps=1000,
)
dqn_gens = gens

# ── CEM-RL ────────────────────────────────────────────────────────────────────
cemrl_all = repeat_and_collect(
    cem_rl,
    env_id=ENV_ID,
    population_size=100,
    generations=gens,
    max_steps=1000,
    dqn_steps_per_gen=3000,
    elite_fraction=0.5,
    alpha=0.0005,
    gamma=0.99,
    buffer_size=100000,
    batch_size=64,
    target_update_steps=500,
    std_init=0.5,
    std_min=0.05,
    eval_episodes=1,
    visualize=False,
    render_every=10,
)
cemrl_gens = gens


# ── Compute mean and std ──────────────────────────────────────────────────────
def compute_mean_std(data):
    return data.mean(axis=0), data.std(axis=0)


agents_mean,    agents_std    = compute_mean_std(agents_all)
hybrid_mean,    hybrid_std    = compute_mean_std(hybrid_all)
actions_mean,   actions_std   = compute_mean_std(actions_all)
qlearning_mean, qlearning_std = compute_mean_std(qlearning_all)
policies_mean,  policies_std  = compute_mean_std(policies_all)
sarsa_mean,     sarsa_std     = compute_mean_std(sarsa_all)
dqn_mean,       dqn_std       = compute_mean_std(dqn_all)
cemrl_mean,     cemrl_std     = compute_mean_std(cemrl_all)


# ── Save results to file ──────────────────────────────────────────────────────
def compute_percent_growth(data_mean, step=15):
    growth = []
    for gen in range(step - 1, len(data_mean), step):
        prev_gen = gen - step
        if prev_gen < 0 or data_mean[prev_gen] == 0:
            pct = 0.0
        else:
            pct = (data_mean[gen] - data_mean[prev_gen]) / abs(data_mean[prev_gen]) * 100
        growth.append((gen + 1, round(float(pct), 2)))
    if len(data_mean) >= 100 and (100, ) not in [(g, ) for g, _ in growth]:
        prev = data_mean[84]
        pct = 0.0 if prev == 0 else (data_mean[99] - prev) / abs(prev) * 100
        growth.append((100, round(float(pct), 2)))
    return growth


results = {
    "env_id": ENV_ID,
    "runs": runs,
    "algorithms": {
        "Population of Agents": {
            "mean": agents_mean.tolist(),
            "std": agents_std.tolist(),
            "gens": agents_gens,
            "growth": compute_percent_growth(agents_mean),
        },
        "Hybrid (Agents + RL)": {
            "mean": hybrid_mean.tolist(),
            "std": hybrid_std.tolist(),
            "gens": hybrid_gens,
            "growth": compute_percent_growth(hybrid_mean),
        },
        "Population of Actions": {
            "mean": actions_mean.tolist(),
            "std": actions_std.tolist(),
            "gens": actions_gens,
            "growth": compute_percent_growth(actions_mean),
        },
        "Q-Learning": {
            "mean": qlearning_mean.tolist(),
            "std": qlearning_std.tolist(),
            "gens": qlearning_gens,
            "growth": compute_percent_growth(qlearning_mean),
        },
        "Population of Policies": {
            "mean": policies_mean.tolist(),
            "std": policies_std.tolist(),
            "gens": policies_gens,
            "growth": compute_percent_growth(policies_mean),
        },
        "SARSA": {
            "mean": sarsa_mean.tolist(),
            "std": sarsa_std.tolist(),
            "gens": sarsa_gens,
            "growth": compute_percent_growth(sarsa_mean),
        },
        "DQN": {
            "mean": dqn_mean.tolist(),
            "std": dqn_std.tolist(),
            "gens": dqn_gens,
            "growth": compute_percent_growth(dqn_mean),
        },
        "CEM-RL": {
            "mean": cemrl_mean.tolist(),
            "std": cemrl_std.tolist(),
            "gens": cemrl_gens,
            "growth": compute_percent_growth(cemrl_mean),
        },
    }
}

save_path = get_next_filename()
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {save_path}")


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_alg(mean, std, gens, label, color, linestyle="-"):
    x = np.arange(1, gens + 1)
    plt.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=2)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)


plt.figure(figsize=(14, 8))
plot_alg(agents_mean,    agents_std,    agents_gens,    "Population of Agents (100 gens)",   "blue")
plot_alg(hybrid_mean,    hybrid_std,    hybrid_gens,    "Hybrid Agents+RL (100 gens)",        "purple", "-.")
plot_alg(actions_mean,   actions_std,   actions_gens,   "Population of Actions (100 gens)",   "green",  "--")
plot_alg(qlearning_mean, qlearning_std, qlearning_gens, "Q-Learning (100 gens)",               "brown",  "--")
plot_alg(policies_mean,  policies_std,  policies_gens,  "Population of Policies (100 gens)",  "gray",   "--")
plot_alg(sarsa_mean,     sarsa_std,     sarsa_gens,     "SARSA (100 gens)",                    "orange", ":")
plot_alg(dqn_mean,       dqn_std,       dqn_gens,       "DQN (100 gens)",                     "red")
plot_alg(cemrl_mean,     cemrl_std,     cemrl_gens,     "CEM-RL (100 gens)",                   "teal")

plt.xlabel("Generation")
plt.ylabel("Average Reward")
plt.title(f"{ENV_ID}: Comparison of Evolutionary and RL Algorithms (with Std Dev)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nPercentage growth every 5 generations ({ENV_ID}):")
for name, data in results["algorithms"].items():
    print(f"{name}: {data['growth']}")