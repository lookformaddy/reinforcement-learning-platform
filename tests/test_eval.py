from rl_platform.bandits import EpsilonGreedyBandit, UCBBandit
from rl_platform.eval_harness import compare_bandits, compare_gridworld, format_table
from rl_platform.gridworld import GridWorld
from rl_platform.tabular import QLearningAgent, SARSAAgent


def test_compare_bandits_reports_all_agents():
    rows = compare_bandits(
        [0.2, 0.5, 0.8],
        [("eps-greedy", EpsilonGreedyBandit, {"epsilon": 0.1}), ("ucb1", UCBBandit, {})],
        steps=200,
        seeds=(0, 1),
    )
    assert [r["agent"] for r in rows] == ["eps-greedy", "ucb1"]
    assert all(r["mean_regret"] is not None and r["mean_regret"] >= 0 for r in rows)
    assert "eps-greedy" in format_table(rows)


def test_compare_gridworld_agents_reach_goal():
    rows = compare_gridworld(
        GridWorld(),
        [("q-learning", QLearningAgent, {}), ("sarsa", SARSAAgent, {})],
        episodes=300,
        seeds=(0,),
    )
    assert all(r["steps_to_goal"] is not None for r in rows)
    assert all(r["mean_reward"] > 0 for r in rows)
