"""Experiment registry: configs, run results, training curves, seeds,
a best-run promotion gate, and versioned JSON model export/import.

The promotion gate is the safety control: a run becomes the champion only
if it beats the current champion *and* clears an absolute metric floor.
That prevents a "best of a bad batch" model from being promoted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class ExperimentConfig:
    name: str
    agent: str  # e.g. "q-learning", "sarsa", "epsilon-greedy"
    hyperparameters: dict
    seed: int
    environment: str = "gridworld-5x5"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    config: ExperimentConfig
    mean_reward: float
    final_regret: float | None = None
    steps_to_goal: float | None = None
    training_curve: list[float] = field(default_factory=list)
    model: dict | None = None  # e.g. a Q-table from agent.to_json()

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentRegistry:
    """In-memory registry with JSON export/import (versioned models)."""

    def __init__(self) -> None:
        self.runs: list[RunResult] = []
        self.champion: RunResult | None = None
        self.model_version = 0

    def record(self, result: RunResult) -> None:
        self.runs.append(result)

    def best_run(self, metric: str = "mean_reward") -> RunResult | None:
        if not self.runs:
            return None
        return max(self.runs, key=lambda r: getattr(r, metric))

    def promote(self, result: RunResult, metric: str = "mean_reward", floor: float | None = None) -> bool:
        """Promote ``result`` to champion iff it beats the champion and
        clears ``floor`` (when given). Returns True on promotion."""
        value = getattr(result, metric)
        if floor is not None and value < floor:
            return False
        if self.champion is not None and getattr(self.champion, metric) >= value:
            return False
        self.champion = result
        self.model_version += 1
        return True

    def export_model(self, result: RunResult) -> str:
        """Versioned JSON export of a run's model + config."""
        payload = {
            "format_version": 1,
            "model_version": self.model_version,
            "config": result.config.to_dict(),
            "model": result.model,
        }
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def import_model(payload: str) -> dict:
        """Parse an exported model; validates the format version."""
        data = json.loads(payload)
        if data.get("format_version") != 1:
            raise ValueError(f"unsupported model format version: {data.get('format_version')}")
        if "model" not in data:
            raise ValueError("exported model payload is missing 'model'")
        return data

    def to_json(self) -> str:
        return json.dumps([r.to_dict() for r in self.runs], sort_keys=True)
