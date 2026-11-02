"""Offline evaluation: step-wise importance sampling (IS).

Given trajectories logged under a *behavior* policy, estimate the value of
a *target* policy without new environment interaction. Step-wise IS weights
each step's reward by the product of target/behavior probability ratios up
to that step — lower variance than weighting the whole trajectory return
by a single end-of-trajectory ratio.

Probabilities of exactly 0 under the behavior policy for an action the
target policy takes are a hard failure (unsupported action): the estimator
raises rather than silently returning a biased number.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoggedStep:
    reward: float
    behavior_prob: float  # prob. of the logged action under the behavior policy
    target_prob: float  # prob. of the logged action under the target policy


@dataclass
class LoggedTrajectory:
    steps: list[LoggedStep] = field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)


def _stepwise_is_return(traj: LoggedTrajectory, gamma: float) -> float:
    weight = 1.0
    estimate = 0.0
    for t, step in enumerate(traj.steps):
        if step.behavior_prob <= 0.0:
            if step.target_prob > 0.0:
                raise ValueError(
                    "target policy takes an action with zero behavior probability; "
                    "step-wise IS is undefined for this trajectory"
                )
            continue  # both zero: action contributes nothing under either policy
        weight *= step.target_prob / step.behavior_prob
        estimate += (gamma**t) * weight * step.reward
    return estimate


def stepwise_importance_sampling(
    trajectories: list[LoggedTrajectory], gamma: float = 1.0
) -> dict[str, float]:
    """Return the IS value estimate plus diagnostics.

    ``ess`` (effective sample size) is reported on the final per-trajectory
    weights: near the trajectory count means low variance, near 1 means the
    estimate is dominated by a single trajectory.
    """
    if not trajectories:
        raise ValueError("need at least one logged trajectory")
    estimates = [_stepwise_is_return(t, gamma) for t in trajectories]
    # Final cumulative weight per trajectory, for the ESS diagnostic.
    final_weights = []
    for traj in trajectories:
        w = 1.0
        for step in traj.steps:
            if step.behavior_prob > 0.0:
                w *= step.target_prob / step.behavior_prob
        final_weights.append(w)
    sum_w = sum(final_weights)
    sum_w2 = sum(w * w for w in final_weights)
    ess = (sum_w * sum_w) / sum_w2 if sum_w2 > 0 else 0.0
    mean = sum(estimates) / len(estimates)
    return {
        "estimate": mean,
        "n_trajectories": float(len(trajectories)),
        "effective_sample_size": ess,
        "behavior_mean_return": sum(t.total_reward for t in trajectories) / len(trajectories),
    }
