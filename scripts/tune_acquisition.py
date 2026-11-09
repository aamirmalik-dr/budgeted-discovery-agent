"""Tune the GP-UCB baseline on the dev family only, then freeze the result.

Grid: lengthscale x beta x number of confirms. Each setting runs on every dev
campaign with five noise streams (runs 100 to 104, disjoint from the
evaluation runs 0 and 1). The setting with the lowest mean normalised regret
wins; ties go to the first setting in grid order. Held-out campaigns are never
touched. Writes configs/tuned_acquisition.yaml and reports/tuning_dev.csv.

Usage: python scripts/tune_acquisition.py   (from the repository root)
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
import yaml

from budgeted_discovery_agent.baselines import TunedParams, run_tuned
from budgeted_discovery_agent.candidates import load_candidates
from budgeted_discovery_agent.environment import load_campaigns
from budgeted_discovery_agent.evaluation import score

LENGTHSCALES = (0.15, 0.25, 0.4)
BETAS = (0.5, 1.0, 2.0)
CONFIRMS = (1, 2, 3)
RUNS = range(100, 105)


def main() -> None:
    cands, mhash = load_candidates(Path("tests/fixtures/candidates.json"))
    specs = load_campaigns(Path("evaluator/campaigns_dev.json"))
    rows = []
    for ls, beta, nc in itertools.product(LENGTHSCALES, BETAS, CONFIRMS):
        params = TunedParams(lengthscale=ls, beta=beta, n_confirm=nc)
        regrets = [
            score(spec, cands, run_tuned(spec, cands, mhash, run, params))["normalized_regret"]
            for spec in specs
            for run in RUNS
        ]
        rows.append(
            {"lengthscale": ls, "beta": beta, "n_confirm": nc, "regret": float(np.mean(regrets))}
        )
        print(rows[-1])
    best = min(rows, key=lambda r: r["regret"])
    Path("reports").mkdir(exist_ok=True)
    with Path("reports/tuning_dev.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows({**r, "regret": round(r["regret"], 4)} for r in rows)
    header = (
        "# Tuned on the dev family only by scripts/tune_acquisition.py (runs 100 to 104).\n"
        f"# Dev mean normalised regret of this setting: {best['regret']:.4f}. Frozen 2026-11-09.\n"
    )
    body = yaml.safe_dump(
        {"lengthscale": best["lengthscale"], "beta": best["beta"], "n_confirm": best["n_confirm"]},
        sort_keys=False,
    )
    Path("configs/tuned_acquisition.yaml").write_text(header + body, encoding="utf-8", newline="\n")
    print("best", best)


if __name__ == "__main__":
    main()
