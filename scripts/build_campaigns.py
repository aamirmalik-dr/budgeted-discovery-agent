"""Write evaluator/campaigns_dev.json and evaluator/campaigns_heldout.json.

The campaign specs hold the hidden response surfaces, screen-fidelity biases,
noise levels and feasibility constraints. They are read by the simulator and
the evaluator only; no tool returns them. Held-out campaigns use surface and
constraint shapes that do not appear in the dev family. Both files were frozen
on 2026-11-09 before any prompt was written; change them only with a version
bump.

Usage: python scripts/build_campaigns.py   (from the repository root)
"""

from __future__ import annotations

import json
from pathlib import Path

COST = {"screen": 2.0, "confirm": 6.0}


def bump(center, height, width):
    return {"center": list(center), "height": height, "width": width}


def halfspace(weights, threshold, reason):
    return {"kind": "halfspace", "weights": list(weights), "threshold": threshold, "reason": reason}


def band(weights, lo, hi, reason):
    return {
        "kind": "band",
        "weights": list(weights),
        "threshold": lo,
        "upper": hi,
        "reason": reason,
    }


def disk(center, radius, reason):
    return {"kind": "disk", "center": list(center), "radius": radius, "reason": reason}


def campaign(cid, family, shape, seed, surface, constraint, budget=36.0):
    return {
        "campaign_id": cid,
        "family": family,
        "shape": shape,
        "budget_units": budget,
        "cost": COST,
        "seed": seed,
        "surface": surface,
        "constraint": constraint,
    }


def surf(bumps, base=0.1, noise=(0.06, 0.02), bias=(), trend=(0.0, 0.0, 0.0)):
    return {
        "base": base,
        "trend": list(trend),
        "bumps": list(bumps),
        "screen_bias": list(bias),
        "noise": {"screen": noise[0], "confirm": noise[1]},
    }


DEV = [
    campaign(
        "dev-smooth",
        "dev",
        "single broad peak, low noise",
        1101,
        surf([bump((0.6, 0.3, 0.1), 0.8, 0.25)], noise=(0.05, 0.02)),
        halfspace(
            (0, 0, 1), 0.7, "prohibited by synthetic stability rule S1: component C above 0.7"
        ),
    ),
    campaign(
        "dev-neartie",
        "dev",
        "two near-equal narrow peaks, noisy screen",
        1102,
        surf(
            [bump((0.2, 0.7, 0.1), 0.80, 0.12), bump((0.7, 0.1, 0.2), 0.76, 0.12)],
            noise=(0.12, 0.03),
        ),
        disk((0.1, 0.1, 0.8), 0.25, "prohibited by synthetic rule S2: C-rich corner is unstable"),
    ),
    campaign(
        "dev-deceptive",
        "dev",
        "screen bias creates a false optimum",
        1103,
        surf(
            [bump((0.2, 0.2, 0.6), 0.7, 0.15), bump((0.7, 0.2, 0.1), 0.3, 0.15)],
            noise=(0.05, 0.03),
            bias=[bump((0.7, 0.2, 0.1), 0.55, 0.12)],
        ),
        halfspace((1, 0, 0), 0.9, "prohibited by synthetic rule S3: component A above 0.9"),
    ),
    campaign(
        "dev-boundary",
        "dev",
        "optimum sits on an infeasible boundary",
        1104,
        surf([bump((0.53, 0.47, 0.0), 0.8, 0.2)], noise=(0.06, 0.02)),
        disk((0.6, 0.4, 0.0), 0.15, "prohibited by synthetic rule S4: A-B eutectic window"),
    ),
]

HELDOUT = [
    campaign(
        "ho-ridge",
        "heldout",
        "ridge of rising bumps, band constraint",
        2201,
        surf([bump((0.8 - 0.14 * i, 0.1, 0.1 + 0.14 * i), 0.4 + 0.07 * i, 0.12) for i in range(6)]),
        band(
            (0, 1, 0),
            0.45,
            0.65,
            "prohibited by synthetic rule H1: component B between 0.45 and 0.65",
        ),
    ),
    campaign(
        "ho-multimodal",
        "heldout",
        "three separated peaks",
        2202,
        surf(
            [
                bump((0.8, 0.1, 0.1), 0.62, 0.1),
                bump((0.1, 0.8, 0.1), 0.70, 0.1),
                bump((0.2, 0.2, 0.6), 0.55, 0.1),
            ],
            noise=(0.08, 0.02),
        ),
        halfspace((1, 1, 0), 0.95, "prohibited by synthetic rule H2: C below 0.05"),
    ),
    campaign(
        "ho-needle",
        "heldout",
        "narrow peak on a broad plateau",
        2203,
        surf([bump((0.6, 0.2, 0.2), 0.2, 0.4), bump((0.2, 0.5, 0.3), 0.45, 0.07)], base=0.35),
        halfspace((0, 0, 1), 0.8, "prohibited by synthetic rule H3: component C above 0.8"),
    ),
    campaign(
        "ho-cornertie",
        "heldout",
        "near-tie between opposite corners, noisy screen",
        2204,
        surf(
            [bump((0.9, 0.0, 0.1), 0.70, 0.15), bump((0.0, 0.9, 0.1), 0.67, 0.15)],
            noise=(0.15, 0.03),
        ),
        disk((0.33, 0.33, 0.34), 0.2, "prohibited by synthetic rule H4: central miscibility gap"),
    ),
    campaign(
        "ho-twobias",
        "heldout",
        "two screen-only false peaks",
        2205,
        surf(
            [bump((0.1, 0.6, 0.3), 0.6, 0.15)],
            noise=(0.05, 0.02),
            bias=[bump((0.6, 0.1, 0.3), 0.7, 0.12), bump((0.3, 0.0, 0.7), 0.6, 0.12)],
        ),
        halfspace((0, 1, 0), 0.85, "prohibited by synthetic rule H5: component B above 0.85"),
    ),
    campaign(
        "ho-trend",
        "heldout",
        "linear trend toward a corner with a small bump",
        2206,
        surf([bump((0.3, 0.6, 0.1), 0.15, 0.15)], base=0.1, trend=(0.2, 0.0, 0.6)),
        halfspace((0, 0, 1), 0.85, "prohibited by synthetic rule H6: component C above 0.85"),
    ),
    campaign(
        "ho-excluded",
        "heldout",
        "global maximum inside the prohibited region",
        2207,
        surf([bump((0.5, 0.0, 0.5), 0.9, 0.12), bump((0.1, 0.8, 0.1), 0.6, 0.15)]),
        disk((0.5, 0.0, 0.5), 0.2, "prohibited by synthetic rule H7: A-C reactive window"),
    ),
    campaign(
        "ho-noisy",
        "heldout",
        "single peak under heavy noise",
        2208,
        surf([bump((0.4, 0.4, 0.2), 0.5, 0.2)], noise=(0.2, 0.06)),
        band(
            (1, 0, 0),
            0.65,
            0.85,
            "prohibited by synthetic rule H8: component A between 0.65 and 0.85",
        ),
    ),
    campaign(
        "ho-lowbudget",
        "heldout",
        "two peaks, tight budget",
        2209,
        surf([bump((0.7, 0.0, 0.3), 0.7, 0.18), bump((0.0, 0.3, 0.7), 0.5, 0.18)]),
        halfspace((0, 1, 0), 0.7, "prohibited by synthetic rule H9: component B above 0.7"),
        budget=22.0,
    ),
    campaign(
        "ho-flat",
        "heldout",
        "shallow landscape with small differences",
        2210,
        surf(
            [bump((0.2, 0.5, 0.3), 0.12, 0.15), bump((0.6, 0.3, 0.1), 0.10, 0.15)],
            base=0.5,
            noise=(0.06, 0.02),
        ),
        disk((0.0, 0.0, 1.0), 0.3, "prohibited by synthetic rule H10: C-rich corner"),
    ),
]


def write(path: Path, family: str, campaigns: list) -> None:
    doc = {"version": f"{family}-v1", "frozen": "2026-11-09", "campaigns": campaigns}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    out = Path("evaluator")
    out.mkdir(exist_ok=True)
    write(out / "campaigns_dev.json", "dev", DEV)
    write(out / "campaigns_heldout.json", "heldout", HELDOUT)
    print("wrote evaluator/campaigns_dev.json and evaluator/campaigns_heldout.json")
