import pytest

from rl_platform.offline import LoggedStep, LoggedTrajectory, stepwise_importance_sampling


def _traj(steps):
    return LoggedTrajectory([LoggedStep(**s) for s in steps])


def test_identical_policies_recover_behavior_return():
    trajs = [
        _traj([{"reward": 1.0, "behavior_prob": 0.5, "target_prob": 0.5},
               {"reward": 2.0, "behavior_prob": 0.5, "target_prob": 0.5}]),
        _traj([{"reward": 3.0, "behavior_prob": 0.5, "target_prob": 0.5}]),
    ]
    out = stepwise_importance_sampling(trajs)
    assert out["estimate"] == pytest.approx(3.0)  # (3 + 3) / 2
    assert out["behavior_mean_return"] == pytest.approx(3.0)


def test_stepwise_weights_only_prefix():
    # Target doubles the first action's probability; second step is untouched
    # by the first ratio beyond its own prefix product (0.5/0.5 = 1).
    trajs = [_traj([{"reward": 1.0, "behavior_prob": 0.5, "target_prob": 1.0},
                    {"reward": 10.0, "behavior_prob": 0.5, "target_prob": 0.5}])]
    out = stepwise_importance_sampling(trajs)
    # step0: 2 * 1 = 2; step1: 2 * 1 * 10 = 20; total 22
    assert out["estimate"] == pytest.approx(22.0)


def test_unsupported_action_raises():
    trajs = [_traj([{"reward": 1.0, "behavior_prob": 0.0, "target_prob": 0.5}])]
    with pytest.raises(ValueError):
        stepwise_importance_sampling(trajs)


def test_empty_trajectories_rejected():
    with pytest.raises(ValueError):
        stepwise_importance_sampling([])
