from rl_platform.registry import ExperimentConfig, ExperimentRegistry, RunResult


def _run(name, reward, seed=0):
    cfg = ExperimentConfig(name=name, agent="q-learning", hyperparameters={"alpha": 0.1}, seed=seed)
    return RunResult(config=cfg, mean_reward=reward, training_curve=[reward])


def test_promotion_gate_blocks_below_floor_and_worse_runs():
    reg = ExperimentRegistry()
    good = _run("good", 8.0)
    bad = _run("bad", 2.0)
    assert reg.promote(good, floor=5.0) is True
    assert reg.champion is good
    assert reg.promote(bad, floor=5.0) is False
    assert reg.champion is good
    assert reg.model_version == 1


def test_promotion_requires_beating_champion():
    reg = ExperimentRegistry()
    reg.promote(_run("a", 8.0))
    assert reg.promote(_run("b", 8.0)) is False  # tie does not promote
    assert reg.promote(_run("c", 9.0)) is True
    assert reg.champion.config.name == "c"


def test_best_run_and_export_import_roundtrip():
    reg = ExperimentRegistry()
    for name, reward in [("a", 5.0), ("b", 9.0), ("c", 7.0)]:
        reg.record(_run(name, reward))
    assert reg.best_run().config.name == "b"
    payload = reg.export_model(reg.best_run())
    data = ExperimentRegistry.import_model(payload)
    assert data["format_version"] == 1
    assert data["config"]["name"] == "b"


def test_import_rejects_bad_format():
    import pytest

    with pytest.raises(ValueError):
        ExperimentRegistry.import_model('{"format_version": 99, "model": {}}')
