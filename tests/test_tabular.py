from rl_platform.gridworld import GridWorld
from rl_platform.rng import SeededRNG
from rl_platform.tabular import QLearningAgent, SARSAAgent, epsilon_schedule, train_tabular


def test_gridworld_wall_and_goal():
    env = GridWorld()
    state, reward, done = env.step((0, 0), 0)  # up from top row: wall
    assert state == (0, 0) and reward == env.wall_reward and not done
    state, reward, done = env.step((4, 3), 3)  # right into goal
    assert state == (4, 4) and reward == env.goal_reward and done


def test_gridworld_obstacle_blocks():
    env = GridWorld()
    state, _, done = env.step((2, 1), 3)  # right into obstacle (2, 2)
    assert state == (2, 1) and not done


def test_epsilon_schedule_decays_to_floor():
    assert epsilon_schedule(0) == 1.0
    assert epsilon_schedule(10_000) == 0.05
    assert epsilon_schedule(1) < epsilon_schedule(0)


def test_qlearning_reaches_goal_greedily():
    env = GridWorld()
    agent = QLearningAgent()
    train_tabular(env, agent, 500, SeededRNG(0))
    state, steps, done = env.reset(), 0, False
    while not done and steps < env.max_steps:
        state, _, done = env.step(state, agent.greedy_action(state))
        steps += 1
    assert done
    assert steps <= 12  # shortest path is 8; allow slack for learned detours


def test_sarsa_reaches_goal_greedily():
    env = GridWorld()
    agent = SARSAAgent()
    train_tabular(env, agent, 500, SeededRNG(0))
    state, steps, done = env.reset(), 0, False
    while not done and steps < env.max_steps:
        state, _, done = env.step(state, agent.greedy_action(state))
        steps += 1
    assert done


def test_q_table_json_roundtrip():
    agent = QLearningAgent()
    train_tabular(GridWorld(), agent, 50, SeededRNG(5))
    restored = QLearningAgent.from_json(agent.to_json())
    assert restored.q == agent.q
