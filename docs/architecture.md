# Architecture

## Diagram (description)

```
                ┌─────────────┐
                │  SeededRNG  │  single seed -> all stochastic choices
                └──────┬──────┘
                       │
        ┌──────────────┼──────────────────┐
        ▼              ▼                  ▼
 ┌─────────────┐ ┌───────────┐    ┌──────────────┐
 │  Bandits    │ │ Gridworld │    │ Logged data  │
 │ eps-greedy  │ │ 5x5, walls│    │ (behavior /  │
 │ UCB1        │ │ obstacles │    │  target probs)│
 │ Thompson    │ │ goal      │    └──────┬───────┘
 └──────┬──────┘ └─────┬─────┘           ▼
        │              │          ┌──────────────┐
        ▼              ▼          │ Offline eval │
 ┌─────────────┐ ┌───────────┐    │ step-wise IS │
 │ run_bandit  │ │ Q-learning│    │ + ESS diag.  │
 │ regret      │ │ SARSA     │    └──────┬───────┘
 └──────┬──────┘ └─────┬─────┘           │
        │              │                 ▼
        └──────────────┼────────► ┌──────────────┐
                       ▼          │   Registry   │
              ┌────────────────┐   │ configs/runs │
              │ Eval harness   │   │ promotion    │
              │ same-seed cmp. │   │ gate, JSON   │
              │ mean reward /  │   │ model I/O    │
              │ regret / steps │   └──────────────┘
              └────────────────┘
                       │
                       ▼ (optional, FastAPI only)
              ┌────────────────┐
              │  service.py    │
              │  /bandits/run  │
              │  /gridworld/   │
              │    train       │
              └────────────────┘
```

## Module contracts

| Module | Key interface | Guarantee |
|---|---|---|
| `rng` | `SeededRNG(seed)` | Same seed → identical sequence, across runs |
| `bandits` | `run_bandit(env, agent, steps, rng)` | Cumulative regret is non-decreasing; regret vs. optimal arm |
| `gridworld` | `step(state, action) -> (state, reward, done)` | Deterministic; walls/obstacles leave state unchanged |
| `tabular` | `train_tabular(env, agent, episodes, rng)` | Q-learning and SARSA share the episode loop; only the update differs |
| `policy` | `policy_iteration(env)` | Returns `(values, policy, iterations)`; policy is stable at return |
| `offline` | `stepwise_importance_sampling(trajs)` | Raises on unsupported actions; reports ESS alongside the estimate |
| `registry` | `ExperimentRegistry.promote(result, floor=...)` | Promotion requires beating champion AND clearing the floor |
| `eval_harness` | `compare_bandits` / `compare_gridworld` | Every agent sees the same seeds; tables render `None` as `-` |

## Evaluation methodology

1. **Same seeds for every agent.** Environment luck is held constant, so
   differences in the table are differences in the agent.
2. **Bandits: cumulative regret.** Mean reward alone hides how much an agent
   paid to learn; regret against the optimal arm is the primary metric.
3. **Gridworld: greedy rollouts after training.** Training uses exploration;
   evaluation turns it off, so the reported steps-to-goal measures the learned
   policy, not the exploration schedule.
4. **Exact baseline.** Policy iteration gives the provably optimal 8-step
   path on the default gridworld; learned agents are judged against it.
5. **Offline: ESS diagnostic.** An IS estimate with ESS near 1 is dominated
   by one trajectory and should not drive a ship decision, whatever its value.

## Extension points

- Stochastic environments: implement `transitions()` with multiple outcomes;
  policy evaluation already consumes distributions.
- Function approximation: replace the Q-dict with a parameterised model behind
  the same `select_action`/`update` interface.
- Persistent registry: swap the in-memory run list for a database; the
  `to_json`/`export_model` payloads are already JSON-shaped.
