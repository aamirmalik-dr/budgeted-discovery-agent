"""The simulated lab: hidden response surfaces, noisy assays and feasibility.

Surface parameters come from evaluator/ and are held inside ``Simulator``.
Tools receive only noisy measurements and the feasibility verdict with its
observable reason; nothing here returns a true value to an agent. The
evaluator calls ``true_value`` only after a campaign has committed its answer.

Noise uses common random numbers: the k-th measurement of a candidate at a
fidelity in a given (campaign, run) draws the same noise in every arm, so
arms are compared on paired noise streams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from .schemas import CampaignBrief, Candidate

FIDELITY_CODE = {"screen": 1, "confirm": 2}


class Bump(BaseModel):
    center: tuple[float, float, float]
    height: float
    width: float


class Surface(BaseModel):
    base: float
    trend: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bumps: list[Bump]
    screen_bias: list[Bump] = []
    noise: dict[str, float]


class Constraint(BaseModel):
    kind: str
    reason: str
    weights: tuple[float, float, float] | None = None
    threshold: float | None = None
    upper: float | None = None
    center: tuple[float, float, float] | None = None
    radius: float | None = None


class CampaignSpec(BaseModel):
    """Full campaign definition from evaluator/ (hidden except the brief)."""

    campaign_id: str
    family: str
    shape: str
    budget_units: float
    cost: dict[str, float]
    seed: int
    surface: Surface
    constraint: Constraint

    def brief(self) -> CampaignBrief:
        """The visible part."""
        return CampaignBrief(
            campaign_id=self.campaign_id,
            family=self.family,  # type: ignore[arg-type]
            budget_units=self.budget_units,
            cost=self.cost,
        )


def load_campaigns(path: Path) -> list[CampaignSpec]:
    """Load a family of campaign specs."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CampaignSpec.model_validate(c) for c in data["campaigns"]]


def _xyz(c: Candidate) -> np.ndarray:
    return np.array([c.a, c.b, c.c])


def _bumps(x: np.ndarray, bumps: list[Bump]) -> float:
    return float(
        sum(
            b.height * np.exp(-np.sum((x - np.array(b.center)) ** 2) / (2 * b.width**2))
            for b in bumps
        )
    )


class Simulator:
    """Holds one hidden campaign and answers assay and feasibility requests."""

    def __init__(self, spec: CampaignSpec, candidates: list[Candidate], run: int) -> None:
        self._spec = spec
        self._run = run
        self._index = {c.candidate_id: i for i, c in enumerate(candidates)}
        self._cands = {c.candidate_id: c for c in candidates}

    def has(self, candidate_id: str) -> bool:
        """True if the candidate exists in the manifest."""
        return candidate_id in self._cands

    def feasibility(self, candidate_id: str) -> tuple[bool, str]:
        """Allowed or prohibited, with the observable reason."""
        k = self._spec.constraint
        x = _xyz(self._cands[candidate_id])
        if k.kind == "halfspace":
            bad = float(np.dot(k.weights, x)) > (k.threshold or 0.0)
        elif k.kind == "band":
            v = float(np.dot(k.weights, x))
            bad = (k.threshold or 0.0) < v < (k.upper or 0.0)
        elif k.kind == "disk":
            bad = float(np.linalg.norm(x - np.array(k.center))) < (k.radius or 0.0)
        else:
            raise ValueError(f"unknown constraint kind {k.kind}")
        return (not bad), (k.reason if bad else "no synthetic stability rule applies")

    def true_value(self, candidate_id: str) -> float:
        """Noise-free confirm-fidelity value. Evaluator use only."""
        s = self._spec.surface
        x = _xyz(self._cands[candidate_id])
        return s.base + float(np.dot(s.trend, x)) + _bumps(x, s.bumps)

    def measure(self, candidate_id: str, fidelity: str, k: int) -> float:
        """One noisy assay; ``k`` is the 0-based count of earlier assays of this pair."""
        s = self._spec.surface
        value = self.true_value(candidate_id)
        if fidelity == "screen":
            value += _bumps(_xyz(self._cands[candidate_id]), s.screen_bias)
        rng = np.random.default_rng(
            [self._spec.seed, self._run, self._index[candidate_id], FIDELITY_CODE[fidelity], k]
        )
        return round(value + float(rng.normal(0.0, s.noise[fidelity])), 4)

    def oracle_summary(self, candidates: list[Candidate]) -> dict[str, Any]:
        """Feasible optimum and mean feasible value. Evaluator use only."""
        feasible = [c.candidate_id for c in candidates if self.feasibility(c.candidate_id)[0]]
        values = {cid: self.true_value(cid) for cid in feasible}
        best = max(values, key=values.__getitem__)
        return {
            "best_id": best,
            "f_star": values[best],
            "f_mean": float(np.mean(list(values.values()))),
            "values": values,
        }
