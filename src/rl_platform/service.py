"""Optional FastAPI service layer.

Deliberately thin: it exposes bandit runs and gridworld training over HTTP
and delegates everything to the stdlib core. If FastAPI is not installed,
importing this module raises ImportError — callers (and tests) must treat
the service as optional. The core package never imports this module.
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover - exercised by the skip test
    raise ImportError("FastAPI is required for the service layer") from exc

from rl_platform.bandits import BernoulliBandit, EpsilonGreedyBandit, run_bandit
from rl_platform.gridworld import GridWorld
from rl_platform.rng import SeededRNG
from rl_platform.tabular import QLearningAgent, train_tabular

app = FastAPI(title="Reinforcement Learning Platform")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/bandits/run")
def bandit_run(payload: dict) -> dict:
    arm_probs = [float(p) for p in payload.get("arm_probs", [0.2, 0.5, 0.8])]
    agent = EpsilonGreedyBandit(n_arms=len(arm_probs), epsilon=float(payload.get("epsilon", 0.1)))
    run = run_bandit(
        BernoulliBandit(arm_probs),
        agent,
        int(payload.get("steps", 500)),
        SeededRNG(int(payload.get("seed", 0))),
    )
    return {"total_reward": run.total_reward, "final_regret": run.final_regret}


@app.post("/gridworld/train")
def gridworld_train(payload: dict) -> dict:
    env = GridWorld()
    agent = QLearningAgent(alpha=float(payload.get("alpha", 0.1)))
    run = train_tabular(env, agent, int(payload.get("episodes", 300)), SeededRNG(int(payload.get("seed", 0))))
    return {"mean_reward": run.mean_reward, "episodes": len(run.episode_rewards)}
