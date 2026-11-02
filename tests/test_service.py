"""Service-layer tests: skipped cleanly when FastAPI is not installed."""

import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not installed; service layer is optional")

from fastapi.testclient import TestClient  # noqa: E402

from rl_platform.service import app  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_bandit_run_endpoint():
    resp = client.post("/bandits/run", json={"arm_probs": [0.2, 0.8], "steps": 100, "seed": 0})
    assert resp.status_code == 200
    assert resp.json()["final_regret"] >= 0
