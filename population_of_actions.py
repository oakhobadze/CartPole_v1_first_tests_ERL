import gymnasium as gym
import numpy as np
import time


class ActionSequence:
    def __init__(self, length=None, max_length=500, min_length=50, n_actions=2):
        self.n_actions = n_actions
        if length is None:
            self.length = np.random.randint(min_length, max_length + 1)
        else:
            self.length = length
        self.sequence = np.random.randint(0, n_actions, size=self.length)

    def mutate(self, max_length=500, min_length=50):
        new_seq = self.sequence.copy()
        num_mutations = np.random.randint(1, 4)
        indices = np.random.choice(len(new_seq), size=num_mutations, replace=False)
        new_seq[indices] = np.random.randint(0, self.n_actions, size=num_mutations)

        if np.random.rand() < 0.3:
            delta = np.random.choice([-1, 1])
            new_length = np.clip(len(new_seq) + delta * np.random.randint(1, 10),
                                 min_length, max_length)
            if new_length > len(new_seq):
                extra = np.random.randint(0, self.n_actions, size=new_length - len(new_seq))
                new_seq = np.concatenate([new_seq, extra])
            else:
                new_seq = new_seq[:new_length]

        mutated = ActionSequence(length=len(new_seq), n_actions=self.n_actions)
        mutated.sequence = new_seq
        return mutated

    @staticmethod
    def crossover(parent1, parent2):
        len1, len2 = len(parent1.sequence), len(parent2.sequence)
        min_len = min(len1, len2)
        if min_len < 2:
            child = ActionSequence(length=len1, n_actions=parent1.n_actions)
            child.sequence = parent1.sequence.copy()
            return child

        cx_point = np.random.randint(1, min_len - 1)
        child_seq = np.concatenate([parent1.sequence[:cx_point], parent2.sequence[cx_point:]])

        child = ActionSequence(length=len(child_seq), n_actions=parent1.n_actions)
        child.sequence = child_seq
        return child


def evaluate_sequence(env, action_seq, penalty_factor=0.001, render=False):
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
        if render:
            env.render()
            time.sleep(0.02)

    return total_reward


def evolutionary_actions(env, population_size=20, generations=20, max_seq_length=500,
                          visualize=False, render_every=5):

    n_actions = env.action_space.n
    min_seq_length = 100

    population = [ActionSequence(max_length=max_seq_length, min_length=min_seq_length, n_actions=n_actions)
                  for _ in range(population_size)]
    avg_per_gen = []
    best_ever_sequence = None
    best_ever_reward = -np.inf

    render_env = None
    if visualize:
        env_name = env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else env.spec.id
        render_env = gym.make(env_name, render_mode="human")

    for gen in range(generations):
        rewards = [evaluate_sequence(env, ind) for ind in population]

        best_idx = np.argmax(rewards)
        best_sequence = population[best_idx]

        if rewards[best_idx] > best_ever_reward:
            best_ever_reward = rewards[best_idx]
            best_ever_sequence = best_sequence
            print(f"  *** New best ever: {best_ever_reward:.2f} ***")

        elite_idx = np.argsort(rewards)[-population_size // 5:]
        elites = [population[i] for i in elite_idx]

        new_population = elites.copy()
        new_population.append(best_ever_sequence)

        current_mutation_scale = max_seq_length * (0.995 ** gen)
        current_mutation_scale = max(current_mutation_scale, min_seq_length)

        while len(new_population) < population_size:
            if np.random.rand() < 0.5:
                p1, p2 = np.random.choice(elites, 2, replace=False)
                child = ActionSequence.crossover(p1, p2)
            else:
                parent = np.random.choice(elites)
                child = parent.mutate(max_length=max_seq_length, min_length=min_seq_length)
            new_population.append(child)

        population = new_population
        avg_per_gen.append(np.mean(rewards))
        print(
            f"Gen {gen}, Best: {max(rewards):.2f}, "
            f"Avg: {np.mean(rewards):.2f}, Median: {np.median(rewards):.2f}, "
            f"Best Seq Length: {len(best_sequence.sequence)}, "
            f"Best Ever: {best_ever_reward:.2f}"
        )

        if visualize and render_env is not None:
            if gen % render_every == 0 or gen == generations - 1:
                print(f"  --> Visualizing best sequence from generation {gen}...")
                evaluate_sequence(render_env, best_ever_sequence, render=True)
                time.sleep(0.5)

    if visualize and render_env is not None:
        render_env.close()

    return avg_per_gen


if __name__ == "__main__":
    #env = gym.make("Acrobot-v1")
    # env = gym.make("LunarLander-v3")
    env = gym.make("CartPole-v1")
    evolutionary_actions(env, population_size=200, generations=200, max_seq_length=1000,
                         visualize=False, render_every=5)
    env.close()