#!/usr/bin/env python3
"""Demo: train bandit + gridworld agents, evaluate offline, print tables.

Run from the project root:  python3 demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rl_platform.bandits import EpsilonGreedyBandit, ThompsonSamplingBandit, UCBBandit
from rl_platform.eval_harness import compare_bandits, compare_gridworld, format_table
from rl_platform.gridworld import GridWorld
from rl_platform.offline import LoggedStep, LoggedTrajectory, stepwise_importance_sampling
from rl_platform.policy import policy_iteration
from rl_platform.registry import ExperimentConfig, ExperimentRegistry, RunResult
from rl_platform.tabular import QLearningAgent, SARSAAgent


def main() -> None:
    print("== Bandits (arms 0.2 / 0.5 / 0.8, 1000 steps, seeds 0-2) ==")
    bandit_rows = compare_bandits(
        [0.2, 0.5, 0.8],
        [
            ("epsilon-greedy", EpsilonGreedyBandit, {"epsilon": 0.1}),
            ("ucb1", UCBBandit, {}),
            ("thompson", ThompsonSamplingBandit, {}),
        ],
        steps=1000,
    )
    print(format_table(bandit_rows))

    print("\n== Gridworld (5x5, 300 episodes, seeds 0-2) ==")
    env = GridWorld()
    grid_rows = compare_gridworld(
        env,
        [("q-learning", QLearningAgent, {}), ("sarsa", SARSAAgent, {})],
        episodes=300,
    )
    print(format_table(grid_rows))

    print("\n== Policy iteration (exact baseline) ==")
    _, policy, iterations = policy_iteration(env)
    print(f"converged in {iterations} iteration(s); start action: {policy[env.start]}")

    print("\n== Offline evaluation (step-wise IS, synthetic logged data) ==")
    trajs = [
        LoggedTrajectory(
            [LoggedStep(reward=1.0, behavior_prob=0.5, target_prob=0.5),
             LoggedStep(reward=2.0, behavior_prob=0.5, target_prob=0.5)]
        )
    ]
    print(stepwise_importance_sampling(trajs))

    print("\n== Experiment registry (promotion gate) ==")
    reg = ExperimentRegistry()
    for row in grid_rows:
        cfg = ExperimentConfig(name=row["agent"], agent=row["agent"], hyperparameters={}, seed=0)
        result = RunResult(config=cfg, mean_reward=row["mean_reward"], steps_to_goal=row["steps_to_goal"])
        print(f"promote {row['agent']}: {reg.promote(result, floor=0.0)}")
    print(f"champion: {reg.champion.config.name if reg.champion else None} "
          f"(model version {reg.model_version})")


if __name__ == "__main__":
    main()
