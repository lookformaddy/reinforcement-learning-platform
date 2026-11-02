from rl_platform.gridworld import GridWorld
from rl_platform.policy import evaluate_policy, improve_policy, policy_iteration


def test_policy_iteration_finds_shortest_path():
    env = GridWorld()
    values, policy, iterations = policy_iteration(env)
    assert iterations >= 1
    # Follow the optimal policy from start; it must reach the goal in 8 steps.
    state, steps, done = env.reset(), 0, False
    while not done and steps < env.max_steps:
        state, _, done = env.step(state, policy[state])
        steps += 1
    assert done and steps == 8


def test_policy_evaluation_goal_value_is_zero():
    env = GridWorld()
    policy = {s: 0 for s in env.states()}
    values = evaluate_policy(env, policy)
    assert values[env.goal] == 0.0
    # States nearer the goal under a sane policy should not be worse than start
    # (checked loosely: start value is finite and negative-or-positive bounded).
    assert -100.0 < values[env.start] < 100.0


def test_greedy_improvement_is_deterministic():
    env = GridWorld()
    values, policy, _ = policy_iteration(env)
    assert improve_policy(env, values) == policy
