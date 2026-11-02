"""Reinforcement Learning Platform: bandits, tabular RL, offline evaluation,
and an experiment registry — all from scratch on the Python standard library.

FastAPI is an *optional* service layer only; every core module runs without it.
"""

from rl_platform.rng import SeededRNG
from rl_platform.bandits import (
    BernoulliBandit,
    EpsilonGreedyBandit,
    ThompsonSamplingBandit,
    UCBBandit,
    run_bandit,
)
from rl_platform.gridworld import GridWorld
from rl_platform.tabular import QLearningAgent, SARSAAgent, train_tabular
from rl_platform.policy import evaluate_policy, improve_policy, policy_iteration
from rl_platform.offline import LoggedTrajectory, stepwise_importance_sampling
from rl_platform.registry import ExperimentConfig, ExperimentRegistry, RunResult
from rl_platform.eval_harness import compare_bandits, compare_gridworld, format_table

__all__ = [
    "SeededRNG",
    "BernoulliBandit",
    "EpsilonGreedyBandit",
    "ThompsonSamplingBandit",
    "UCBBandit",
    "run_bandit",
    "GridWorld",
    "QLearningAgent",
    "SARSAAgent",
    "train_tabular",
    "evaluate_policy",
    "improve_policy",
    "policy_iteration",
    "LoggedTrajectory",
    "stepwise_importance_sampling",
    "ExperimentConfig",
    "ExperimentRegistry",
    "RunResult",
    "compare_bandits",
    "compare_gridworld",
    "format_table",
]
