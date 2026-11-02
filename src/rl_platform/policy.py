"""Policy evaluation + improvement (policy iteration) for the gridworld.

Iterative policy evaluation sweeps the Bellman expectation equation until
the max value change falls below ``theta``; greedy improvement then picks
the best action per state. The loop repeats until the policy is stable —
a classic, exact baseline that learned agents can be checked against.
"""

from __future__ import annotations

from rl_platform.gridworld import GridWorld

State = tuple[int, int]


def evaluate_policy(
    env: GridWorld,
    policy: dict[State, int],
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_sweeps: int = 10_000,
) -> dict[State, float]:
    """Value function V for a fixed deterministic policy."""
    values: dict[State, float] = {s: 0.0 for s in env.states()}
    for _ in range(max_sweeps):
        delta = 0.0
        for state in env.states():
            if state == env.goal:
                continue
            action = policy[state]
            new_value = 0.0
            for next_state, reward, done in env.transitions(state, action):
                continuation = 0.0 if done else gamma * values[next_state]
                new_value += reward + continuation
            delta = max(delta, abs(new_value - values[state]))
            values[state] = new_value
        if delta < theta:
            break
    return values


def improve_policy(
    env: GridWorld, values: dict[State, float], gamma: float = 0.99
) -> dict[State, int]:
    """Greedy policy w.r.t. a value function (ties -> lowest action index)."""
    policy: dict[State, int] = {}
    for state in env.states():
        if state == env.goal:
            policy[state] = 0
            continue
        best_action, best_value = 0, float("-inf")
        for action in range(4):
            value = 0.0
            for next_state, reward, done in env.transitions(state, action):
                continuation = 0.0 if done else gamma * values[next_state]
                value += reward + continuation
            if value > best_value:
                best_value, best_action = value, action
        policy[state] = best_action
    return policy


def policy_iteration(
    env: GridWorld, gamma: float = 0.99, theta: float = 1e-6
) -> tuple[dict[State, float], dict[State, int], int]:
    """Alternate evaluation/improvement until stable.

    Returns ``(values, policy, iterations)``.
    """
    policy = {s: 0 for s in env.states()}
    iterations = 0
    while True:
        iterations += 1
        values = evaluate_policy(env, policy, gamma=gamma, theta=theta)
        new_policy = improve_policy(env, values, gamma=gamma)
        if new_policy == policy:
            return values, policy, iterations
        policy = new_policy
