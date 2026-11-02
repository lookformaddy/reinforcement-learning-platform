"""Eval harness: compare agents on the *same* seeded environments.

Same seeds for every agent is the whole point — it removes environment
luck from the comparison. Reports mean reward, (bandit) regret, and
(gridworld) steps-to-goal, and renders a plain-text comparison table.
"""

from __future__ import annotations

from rl_platform.bandits import BernoulliBandit, run_bandit
from rl_platform.gridworld import GridWorld
from rl_platform.rng import SeededRNG
from rl_platform.tabular import TabularRun, train_tabular


def _fresh_agent(factory, kwargs: dict):
    return factory(**kwargs)


def compare_bandits(
    arm_probs: list[float],
    agent_specs: list[tuple[str, object, dict]],
    steps: int = 1000,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict]:
    """``agent_specs``: ``(name, agent_class, kwargs)`` triples."""
    env = BernoulliBandit(arm_probs)
    rows = []
    for name, cls, kwargs in agent_specs:
        rewards, regrets = [], []
        for seed in seeds:
            agent = cls(n_arms=env.n_arms, **kwargs)
            run = run_bandit(env, agent, steps, SeededRNG(seed))
            rewards.append(run.total_reward)
            regrets.append(run.final_regret)
        rows.append(
            {
                "agent": name,
                "mean_reward": sum(rewards) / len(rewards),
                "mean_regret": sum(regrets) / len(regrets),
                "steps_to_goal": None,
            }
        )
    return rows


def compare_gridworld(
    env: GridWorld,
    agent_specs: list[tuple[str, object, dict]],
    episodes: int = 300,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict]:
    """Train each agent from scratch per seed, then measure greedy rollouts."""
    rows = []
    for name, cls, kwargs in agent_specs:
        rewards, lengths = [], []
        for seed in seeds:
            agent = cls(**kwargs)
            train_tabular(env, agent, episodes, SeededRNG(seed))
            # Greedy evaluation rollout (no exploration).
            state = env.reset()
            total, steps, done = 0.0, 0, False
            while not done and steps < env.max_steps:
                action = agent.greedy_action(state)
                state, reward, done = env.step(state, action)
                total += reward
                steps += 1
            rewards.append(total)
            lengths.append(steps if done else None)
        finished = [s for s in lengths if s is not None]
        rows.append(
            {
                "agent": name,
                "mean_reward": sum(rewards) / len(rewards),
                "mean_regret": None,
                "steps_to_goal": (sum(finished) / len(finished)) if finished else None,
            }
        )
    return rows


def format_table(rows: list[dict]) -> str:
    """Plain-text table; ``None`` renders as ``-`` so mixed rows stay aligned."""
    header = f"{'agent':<24} {'mean_reward':>12} {'mean_regret':>12} {'steps_to_goal':>14}"
    lines = [header, "-" * len(header)]
    for row in rows:
        regret = f"{row['mean_regret']:.2f}" if row["mean_regret"] is not None else "-"
        steps = f"{row['steps_to_goal']:.1f}" if row["steps_to_goal"] is not None else "-"
        lines.append(
            f"{row['agent']:<24} {row['mean_reward']:>12.2f} {regret:>12} {steps:>14}"
        )
    return "\n".join(lines)
