import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob


def load_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def plot_from_file(filepath):
    data = load_results(filepath)
    env_id = data["env_id"]

    colors = {
        "Population of Agents":   ("blue",      "-"),
        "Hybrid (Agents + RL)":   ("purple",    "-."),
        "Population of Actions":  ("green",     "--"),
        "Q-Learning":             ("brown",     "--"),
        "Population of Policies": ("gray",      "--"),
        "SARSA":                  ("orange",    ":"),
        "DQN":                    ("red",       "-"),
        "CEM-RL":                 ("teal",      "-"),
    }

    plt.figure(figsize=(14, 8))

    for name, alg_data in data["algorithms"].items():
        mean = np.array(alg_data["mean"])
        std = np.array(alg_data["std"])

        if len(mean) == 0:
            continue

        gens = len(mean)

        color, linestyle = colors.get(name, ("black", "-"))
        x = np.arange(1, gens + 1)
        plt.plot(x, mean, label=f"{name} ({gens} gens)",
                 color=color, linestyle=linestyle, linewidth=2)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    plt.xlabel("Generation")
    plt.ylabel("Average Reward")
    plt.title(f"{env_id}: Comparison of Evolutionary and RL Algorithms (with Std Dev)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_growth_table(filepath):
    data = load_results(filepath)
    env_id = data["env_id"]

    print(f"\nPercentage growth every 5 generations ({env_id}):")
    for name, alg_data in data["algorithms"].items():
        growth = alg_data.get("growth")
        if growth is not None:
            print(f"{name}: {growth}")
        else:
            print(f"{name}: no growth data available")


def print_latex_table(filepath):
    data = load_results(filepath)
    env_id = data["env_id"]

    checkpoints = [0, 15, 30, 45, 60, 75, 90, 100]

    print(f"\n% LaTeX table for {env_id}")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{@{}lccccccccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Algorithm} & " +
          " & ".join([f"\\textbf{{Gen {g}}}" for g in checkpoints]) +
          r" & \textbf{Std Dev} \\ \midrule")

    for name, alg_data in data["algorithms"].items():
        mean = np.array(alg_data["mean"])
        std = np.array(alg_data["std"])
        std_range = f"{round(float(np.min(std)), 1)}--{round(float(np.max(std)), 1)}"

        growth = alg_data.get("growth")
        if growth is not None:
            growth_dict = {gen: pct for gen, pct in growth}
            row_values = []
            for g in checkpoints:
                if g in growth_dict:
                    row_values.append(f"{growth_dict[g]:.2f}\\%")
                else:
                    row_values.append("--")
        else:
            row_values = ["--"] * len(checkpoints)

        print(f"{name} & " + " & ".join(row_values) + f" & {std_range} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(f"\\caption{{Percentage growth every 15 generations and approximate "
          f"standard deviation for each algorithm on {env_id}}}")
    print(f"\\label{{tab:metrics_{env_id.lower().replace('-', '_').replace('.', '_')}}}")
    print(r"\end{table}")


def list_available_results(base_dir="results"):
    files = sorted(glob.glob(os.path.join(base_dir, "test*.json")))
    if not files:
        print("No result files found in results/ directory.")
        return []
    print("Available result files:")
    for i, f in enumerate(files):
        data = load_results(f)
        print(f"  [{i+1}] {f} — env: {data['env_id']}, runs: {data['runs']}")
    return files


if __name__ == "__main__":
    files = list_available_results()
    if not files:
        exit()

    choice = input("\nEnter file number to visualize (or press Enter for latest): ").strip()
    if choice == "":
        filepath = files[-1]
    else:
        filepath = files[int(choice) - 1]

    print(f"\nLoading: {filepath}")

    plot_from_file(filepath)
    print_growth_table(filepath)
    print_latex_table(filepath)