# Reinforcement Learning Platform

An enterprise-grade, from-scratch reinforcement learning toolkit: multi-armed
bandits, tabular agents on a gridworld, exact policy iteration, offline
policy evaluation, and an experiment registry with a promotion gate — all in
pure Python standard library, with FastAPI only as an optional service layer.

## The enterprise problem

Production RL decisions — which experiment to run, which policy to ship, how
to evaluate a new policy without new traffic — are usually made on vibes.
This platform makes each of those decisions measurable:

- **Bandits** for low-risk online allocation (which variant gets traffic).
- **Tabular agents** (Q-learning, SARSA) as the teaching and baseline layer
  for sequential decision problems.
- **Policy iteration** as an exact baseline: learned agents can be checked
  against the provably optimal policy on small environments.
- **Offline evaluation** (step-wise importance sampling) to estimate a new
  policy's value from logged behavior data — before it ever touches users.
- **Experiment registry** so runs are reproducible (seed + config + curve)
  and only runs that beat the champion *and* clear a metric floor get promoted.

## Architecture overview

```
src/rl_platform/
├── rng.py          # SeededRNG: one seed reproduces any experiment
├── bandits.py      # BernoulliBandit, epsilon-greedy, UCB1, Thompson sampling
├── gridworld.py    # Deterministic 5x5 gridworld (obstacles, walls, goal)
├── tabular.py      # Q-learning, SARSA, epsilon decay schedules
├── policy.py       # Iterative policy evaluation + greedy improvement loop
├── offline.py      # Step-wise importance sampling estimator + ESS diagnostic
├── registry.py     # Experiment configs, runs, promotion gate, JSON model I/O
├── eval_harness.py # Same-seed agent comparison + comparison tables
└── service.py      # OPTIONAL FastAPI layer (requires fastapi; core never imports it)
demo.py             # Trains bandits + gridworld agents, prints comparison tables
tests/              # pytest coverage of every core module
```

See [docs/architecture.md](docs/architecture.md) for module contracts, the
evaluation methodology, and a diagram description.

## How to run

```bash
pip install -r requirements.txt   # pytest only; fastapi optional
python3 demo.py                   # bandit + gridworld comparison tables
pytest                            # full test suite (service tests skip without FastAPI)
```

Optional service (requires FastAPI):

```bash
pip install fastapi uvicorn
uvicorn rl_platform.service:app --reload  # from src/, or set PYTHONPATH=src
```

## Key design decisions and trade-offs

- **Stdlib core, zero ML dependencies.** No gym, no sklearn, no numpy. The
  algorithms are the product here; implementing them from scratch keeps the
  code auditable, dependency-free, and runnable anywhere Python runs. The
  cost is speed on large problems — this is a platform for correct baselines
  and evaluation methodology, not for Atari-scale training.
- **One seed, one experiment.** All randomness flows through `SeededRNG`,
  so a run is fully described by `(config, seed)`. No global RNG state.
- **Deterministic gridworld, distribution-shaped API.** `GridWorld.transitions`
  returns a one-element distribution, so policy evaluation code works unchanged
  if the environment later becomes stochastic.
- **Step-wise IS, not whole-trajectory IS.** Weighting each step's reward by
  its prefix ratio cuts variance sharply. The estimator *raises* on
  unsupported actions (target prob > 0, behavior prob = 0) instead of silently
  returning a biased estimate.
- **Promotion gate, not "best run wins".** A run promotes only if it beats the
  champion and clears an absolute floor — preventing a best-of-a-bad-batch
  model from shipping.
- **Q-learning vs SARSA.** Q-learning (off-policy, max bootstrap) learns the
  optimal policy faster here; SARSA (on-policy) is more conservative near
  costly states. Both are shipped so the difference is measurable, not argued.

## What it demonstrates

Bandit regret analysis, on- vs off-policy TD learning, exact dynamic
programming baselines, offline policy evaluation with variance diagnostics,
and the registry/promotion discipline that production ML platforms require —
all reproducible from a single seed.

## CI note

The CI workflow lives at `docs/github-actions-ci.yml`. The GitHub App used
for pushing lacks the Workflows permission, so `.github/workflows/` is not
shipped in this repo. To activate CI, move that file to
`.github/workflows/ci.yml` after the first push.

## Clean-room note

All code is original, all data is synthetic (fixed Bernoulli arm
probabilities, a hand-built gridworld). No employer code, data, internal
architecture, project names, or proprietary designs are used.
