"""Write evaluator/campaigns_heldout_v2.json, the second held-out family.

Written on 2026-11-09 after the pre-registered v2 fixes were committed and
before any v2 run. Surface shapes (central valley, plateau with an edge spike,
elongated chain, four peaks, a negative screen bias, a peak behind a wedge,
broad noisy peak with a larger budget, clustered positive bias, steep trend,
small-budget near-tie) and constraint shapes (two disks, wedge, ring) are new
relative to the dev and v1 held-out families. No prompt, fix or tuning has seen
these campaigns. The v1 files are not touched.

Usage: python scripts/build_campaigns_v2.py   (from the repository root)
"""

from __future__ import annotations

import json
from pathlib import Path

from build_campaigns import band, bump, campaign, halfspace, surf


def two_disks(centers, radius, reason):
    return {
        "kind": "two_disks",
        "centers": [list(c) for c in centers],
        "radius": radius,
        "reason": reason,
    }


def wedge(w1, t1, w2, t2, reason):
    return {"kind": "wedge", "weights": list(w1), "threshold": t1, "weights2": list(w2), "threshold2": t2,
            "reason": reason}  # fmt: skip


def ring(center, inner, outer, reason):
    return {
        "kind": "ring",
        "center": list(center),
        "radius": inner,
        "outer": outer,
        "reason": reason,
    }


F = "heldout_v2"
CENTRE = (1 / 3, 1 / 3, 1 / 3)
HELDOUT_V2 = [
    campaign(
        "h2-valley",
        F,
        "two edge peaks around a central valley",
        3301,
        surf(
            [
                bump((0.8, 0.2, 0.0), 0.55, 0.15),
                bump((0.0, 0.2, 0.8), 0.62, 0.15),
                bump(CENTRE, -0.2, 0.2),
            ],
            base=0.3,
        ),
        two_disks(
            [(0.6, 0.0, 0.4), (0.0, 0.6, 0.4)],
            0.15,
            "prohibited by synthetic rule V1: two unstable pockets",
        ),
    ),
    campaign(
        "h2-plateau-edge",
        F,
        "broad plateau with a spike at its edge",
        3302,
        surf([bump((0.4, 0.4, 0.2), 0.5, 0.35), bump((0.6, 0.4, 0.0), 0.25, 0.08)], base=0.2),
        wedge(
            (1, 0, 0),
            0.75,
            (0, 0, -1),
            -0.15,
            "prohibited by synthetic rule V2: A above 0.75 with C below 0.15",
        ),
    ),
    campaign(
        "h2-chain",
        F,
        "elongated chain of bumps along B",
        3303,
        surf(
            [
                bump((0.1, 0.2 + 0.15 * i, 0.7 - 0.15 * i), h, 0.1)
                for i, h in enumerate((0.5, 0.55, 0.62, 0.58))
            ]
        ),
        ring(CENTRE, 0.25, 0.32, "prohibited by synthetic rule V3: ring-shaped miscibility gap"),
    ),
    campaign(
        "h2-fourpeak",
        F,
        "four small separated peaks",
        3304,
        surf(
            [
                bump((0.7, 0.1, 0.2), 0.55, 0.09),
                bump((0.1, 0.7, 0.2), 0.6, 0.09),
                bump((0.2, 0.1, 0.7), 0.66, 0.09),
                bump((0.4, 0.4, 0.2), 0.5, 0.09),
            ],
            noise=(0.07, 0.02),
        ),
        halfspace((0, 0, 1), 0.75, "prohibited by synthetic rule V4: component C above 0.75"),
    ),
    campaign(
        "h2-negbias",
        F,
        "screen under-reports the true optimum",
        3305,
        surf(
            [bump((0.6, 0.1, 0.3), 0.7, 0.15), bump((0.2, 0.6, 0.2), 0.55, 0.15)],
            noise=(0.05, 0.02),
            bias=[bump((0.6, 0.1, 0.3), -0.3, 0.1)],
        ),
        two_disks(
            [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)],
            0.22,
            "prohibited by synthetic rule V5: pure B and pure C corners",
        ),
    ),
    campaign(
        "h2-wedge-peak",
        F,
        "largest peak sits behind a wedge-shaped rule",
        3306,
        surf([bump((0.9, 0.1, 0.0), 0.85, 0.12), bump((0.5, 0.1, 0.4), 0.6, 0.15)]),
        wedge(
            (1, 0, 0),
            0.65,
            (0, 0, -1),
            -0.25,
            "prohibited by synthetic rule V6: A above 0.65 with C below 0.25",
        ),
    ),
    campaign(
        "h2-broad-noisy",
        F,
        "broad peak, heavy screen noise, larger budget",
        3307,
        surf(
            [bump((0.3, 0.5, 0.2), 0.45, 0.3), bump((0.25, 0.55, 0.2), 0.1, 0.06)],
            noise=(0.18, 0.05),
        ),
        band((0, 0, 1), 0.45, 0.62, "prohibited by synthetic rule V7: C between 0.45 and 0.62"),
        budget=48.0,
    ),
    campaign(
        "h2-cluster-bias",
        F,
        "positive screen bias on a cluster away from the optimum",
        3308,
        surf(
            [bump((0.3, 0.0, 0.7), 0.6, 0.15), bump((0.6, 0.3, 0.1), 0.3, 0.2)],
            noise=(0.05, 0.02),
            bias=[bump((0.7, 0.3, 0.0), 0.35, 0.1), bump((0.6, 0.4, 0.0), 0.35, 0.1)],
        ),
        two_disks(
            [(0.0, 0.5, 0.5), (0.5, 0.5, 0.0)],
            0.12,
            "prohibited by synthetic rule V8: two eutectic pockets",
        ),
    ),
    campaign(
        "h2-steep",
        F,
        "steep trend toward A cut by a rule, plus a minor bump",
        3309,
        surf([bump((0.2, 0.2, 0.6), 0.3, 0.12)], base=0.1, trend=(0.7, 0.25, 0.0)),
        halfspace((1, 0, 0), 0.8, "prohibited by synthetic rule V9: component A above 0.8"),
    ),
    campaign(
        "h2-tight-tie",
        F,
        "near-tie between edge peaks on a small budget",
        3310,
        surf(
            [bump((0.5, 0.5, 0.0), 0.6, 0.12), bump((0.0, 0.5, 0.5), 0.57, 0.12)], noise=(0.1, 0.02)
        ),
        ring(CENTRE, 0.12, 0.2, "prohibited by synthetic rule V10: inner ring"),
        budget=26.0,
    ),
]


if __name__ == "__main__":
    doc = {"version": "heldout_v2-v1", "frozen": "2026-11-09", "campaigns": HELDOUT_V2}
    Path("evaluator/campaigns_heldout_v2.json").write_text(
        json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("wrote evaluator/campaigns_heldout_v2.json")
