import re
import matplotlib.pyplot as plt
import numpy as np


def parse_results(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    known_headers = {
        'population_of_agents',
        'hybrid',
        'dqn',
        'qlearning',
        'sarsa',
        'population of policies',
        'population of actions',
        'cem-rl',
    }

    header_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped in known_headers:
            header_lines.append((i, stripped))

    header_lines.sort()

    sections = {}
    for idx, (start_line, name) in enumerate(header_lines):
        end_line = header_lines[idx + 1][0] if idx + 1 < len(header_lines) else len(lines)
        sections[name] = lines[start_line:end_line]

    results = {}

    # ── Population of Agents ──────────────────────────────────────────────────
    alg_lines = sections.get('population_of_agents', [])
    bests, medians = [], []
    for line in alg_lines:
        m = re.search(r'Generation\s+\d+,\s*Best:\s*([-\d.]+).*?Median:\s*([-\d.]+)', line)
        if m:
            bests.append(float(m.group(1)))
            medians.append(float(m.group(2)))
    if bests:
        results['Population of Agents\n(Acrobot-v1)'] = {'best': bests, 'median': medians}

    # ── Hybrid ────────────────────────────────────────────────────────────────
    alg_lines = sections.get('hybrid', [])
    bests, medians = [], []
    for line in alg_lines:
        m = re.search(r'Best:\s*([-\d.]+).*?Median:\s*([-\d.]+)', line)
        if m and 'Training' not in line:
            bests.append(float(m.group(1)))
            medians.append(float(m.group(2)))
    if bests:
        results['Hybrid (Agents+RL)\n(LunarLander-v3)'] = {'best': bests, 'median': medians}

    # ── DQN ───────────────────────────────────────────────────────────────────
    alg_lines = sections.get('dqn', [])
    bests, medians = [], []
    for line in alg_lines:
        m = re.search(r'Generation\s+\d+,\s*Best:\s*([-\d.]+).*?Median:\s*([-\d.]+)', line)
        if m:
            bests.append(float(m.group(1)))
            medians.append(float(m.group(2)))
    if bests:
        results['DQN\n(LunarLander-v3)'] = {'best': bests, 'median': medians}

    # ── CEM-RL ────────────────────────────────────────────────────────────────
    alg_lines = sections.get('cem-rl', [])
    bests, medians = [], []
    for line in alg_lines:
        m = re.search(r'Generation\s+\d+,\s*Best:\s*([-\d.]+).*?Median:\s*([-\d.]+)', line)
        if m:
            bests.append(float(m.group(1)))
            medians.append(float(m.group(2)))
    if bests:
        results['CEM-RL\n(LunarLander-v3)'] = {'best': bests, 'median': medians}

    # ── Q-Learning ────────────────────────────────────────────────────────────
    alg_lines = sections.get('qlearning', [])
    bests, medians = [], []
    for line in alg_lines:
        m = re.search(r'Gen\s+\d+,\s*Best:\s*([-\d.]+).*?Median:\s*([-\d.]+)', line)
        if m:
            bests.append(float(m.group(1)))
            medians.append(float(m.group(2)))
    if bests:
        results['Q-Learning\n(Acrobot-v1)'] = {'best': bests, 'median': medians}

    # ── SARSA ─────────────────────────────────────────────────────────────────
    alg_lines = sections.get('sarsa', [])
    bests, medians = [], []
    for line in alg_lines:
        m = re.search(r'Generation\s+\d+,\s*Best:\s*([-\d.]+).*?Median:\s*([-\d.]+)', line)
        if m:
            bests.append(float(m.group(1)))
            medians.append(float(m.group(2)))
    if bests:
        results['SARSA\n(Acrobot-v1)'] = {'best': bests, 'median': medians}

    # ── Population of Policies ────────────────────────────────────────────────
    alg_lines = sections.get('population of policies', [])
    bests, medians = [], []
    for line in alg_lines:
        m = re.search(r'Gen\s+\d+,\s*Best:\s*([-\d.]+).*?Median:\s*([-\d.]+)', line)
        if m:
            bests.append(float(m.group(1)))
            medians.append(float(m.group(2)))
    if bests:
        results['Population of Policies\n(CartPole-v1)'] = {'best': bests, 'median': medians}

    # ── Population of Actions ─────────────────────────────────────────────────
    alg_lines = sections.get('population of actions', [])
    bests, medians = [], []
    for line in alg_lines:
        m = re.search(r'Gen\s+\d+,\s*Best:\s*([-\d.]+).*?Median:\s*([-\d.]+)', line)
        if m:
            bests.append(float(m.group(1)))
            medians.append(float(m.group(2)))
    if bests:
        results['Population of Actions\n(CartPole-v1)'] = {'best': bests, 'median': medians}

    return results


def plot_finetune(results):
    colors = {
        'Population of Agents\n(Acrobot-v1)':       'blue',
        'Hybrid (Agents+RL)\n(LunarLander-v3)':     'purple',
        'DQN\n(LunarLander-v3)':                    'red',
        'CEM-RL\n(LunarLander-v3)':                 'teal',
        'Q-Learning\n(Acrobot-v1)':                 'brown',
        'SARSA\n(Acrobot-v1)':                      'orange',
        'Population of Policies\n(CartPole-v1)':    'gray',
        'Population of Actions\n(CartPole-v1)':     'green',
    }

    items = list(results.items())

    def plot_single(ax, name, data):
        best = np.array(data['best'])
        median = np.array(data['median'])
        x = np.arange(1, len(best) + 1)
        color = colors.get(name, 'black')
        ax.plot(x, best, label='Best per gen', color=color, linewidth=2)
        ax.plot(x, median, label='Median per gen', color=color,
                linewidth=1.5, linestyle='--', alpha=0.7)
        ax.fill_between(x, median, best, alpha=0.15, color=color)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Reward')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── Page 1: 2x2 grid (Population of Agents, Hybrid, DQN, CEM-RL) ─────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, (name, data) in zip(axes, items[:4]):
        plot_single(ax, name, data)

    plt.suptitle('Fine-Tuned Algorithm Performance\n(Best Agent vs Median per Generation)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('finetune_page1.png', dpi=150, bbox_inches='tight')
    print("Saved: finetune_page1.png")
    plt.show()

    # ── Page 2: 2x2 grid (Q-Learning, SARSA, Population of Policies, Population of Actions)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    for ax, (name, data) in zip(axes, items[4:]):
        plot_single(ax, name, data)

    plt.suptitle('Fine-Tuned Algorithm Performance\n(Best Agent vs Median per Generation)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('finetune_page2.png', dpi=150, bbox_inches='tight')
    print("Saved: finetune_page2.png")
    plt.show()


def print_summary(results):
    print("\n" + "="*70)
    print(f"{'Algorithm':<40} {'Gens':>6} {'Final Best':>12} {'Final Median':>13}")
    print("="*70)
    for name, data in results.items():
        label = name.replace('\n', ' ')
        gens = len(data['best'])
        final_best = data['best'][-1]
        final_median = data['median'][-1]
        best_ever = max(data['best'])
        print(f"{label:<40} {gens:>6} {final_best:>12.2f} {final_median:>13.2f}  (best ever: {best_ever:.2f})")
    print("="*70)


if __name__ == "__main__":
    filepath = "results_of_fine_tune.txt"

    print(f"Parsing: {filepath}")
    results = parse_results(filepath)

    print(f"\nFound {len(results)} algorithms:")
    for name, data in results.items():
        print(f"  {name.replace(chr(10), ' ')}: {len(data['best'])} generations")

    print_summary(results)
    plot_finetune(results)