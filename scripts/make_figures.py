"""Draw the README figures from saved run outputs only.

Reads reports/live (v1, gpt-4.1-mini), reports/live_v2 (v2 run A,
gpt-4.1-mini) and reports/live_v2_sonnet (v2 run B, second model):
task_results.jsonl, the per-campaign receipt ledgers and, for the example
campaign, the saved agent trace. True values for the anytime-regret curves and
the example campaign come from the frozen campaign definitions in evaluator/
through the deterministic simulator, read after the fact as the scorer does.
No model is called and nothing is estimated. Running this script twice gives
identical files. The aborted run B attempt (reports/aborted/) is not drawn.

Because this repository already carries large saved outputs, each PNG is
re-encoded after saving with a 256-colour palette built from its own most
frequent colours: the hardcoded fills, lines and text keep their exact
colours, and only rare anti-aliasing shades map to the nearest kept colour.

Usage: python scripts/make_figures.py [--out docs/figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from budgeted_discovery_agent.candidates import load_candidates  # noqa: E402
from budgeted_discovery_agent.environment import Simulator, load_campaigns  # noqa: E402

# (output dir, campaign family, group label, second model?)
ROUNDS = [
    ("live", "heldout", "v1, gpt-4.1-mini\nheld-out v1", False),
    ("live_v2", "heldout_v2", "v2, gpt-4.1-mini\nheld-out v2", False),
    ("live_v2_sonnet", "heldout_v2", "v2, second model\nheld-out v2", True),
]
SAMPLE = (
    "Live. v1 and v2: gpt-4.1-mini, 10 campaigns x 2 noise streams.\n"
    "Second model: claude-sonnet-4.6, 10 campaigns x 1 stream, agent hatched."
)
ARMS = [
    "random_feasible",
    "cost_aware",
    "tuned_acquisition",
    "single_call",
    "agent",
    "agent_no_replicate",
    "agent_default_selector",
]
ARM_TYPE = {
    "random_feasible": "baseline",
    "cost_aware": "baseline",
    "tuned_acquisition": "baseline",
    "single_call": "single",
    "agent": "agent",
    "agent_no_replicate": "ablation",
    "agent_default_selector": "ablation",
}
ARM_LABEL = {
    "random_feasible": "random",
    "cost_aware": "cost-aware",
    "tuned_acquisition": "tuned UCB",
    "single_call": "single",
    "agent": "agent",
    "agent_no_replicate": "no-rep",
    "agent_default_selector": "def-sel",
}
KEY = (
    "random, cost-aware, tuned UCB: no model.  single: one model call sets the allocation.\n"
    "agent: iterative agent.  no-rep: agent without replicate.  def-sel: proposals forced "
    "to balanced."
)
PENDING = "run_budget_exhausted"
FIGURES = ["regret_by_arm.png", "regret_vs_spend.png", "stop_reasons.png", "example_campaign.png"]


def results(name: str) -> list[dict]:
    path = ROOT / "reports" / name / "task_results.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def by_arm(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["arm"]].append(r)
    return out


def layout(arms: list[str]) -> list[tuple[float, str, bool, str]]:
    """Bar positions: (x, round dir, second model, arm)."""
    pos, x = [], 0.0
    for name, _, _, second in ROUNDS:
        present = by_arm(results(name))
        for arm in arms:
            if arm in present:
                pos.append((x, name, second, arm))
                x += 1
        x += 0.9
    return pos


def label_groups(ax, pos, y: float = -0.2) -> None:
    """Arm names as rotated ticks, round names centred under each group."""
    ax.set_xticks([p[0] for p in pos], [ARM_LABEL[p[3]] for p in pos], rotation=30, ha="right")
    for name, _, title, _ in ROUNDS:
        xs = [p[0] for p in pos if p[1] == name]
        if xs:
            ax.text(
                sum(xs) / len(xs),
                y,
                title,
                ha="center",
                va="top",
                fontsize=10,
                transform=ax.get_xaxis_transform(),
            )
    ax.set_xlim(pos[0][0] - 0.7, pos[-1][0] + 1.0)


# ---------------------------------------------------------------- figure 1


def regret_by_arm(out: Path) -> None:
    pos = layout(ARMS)
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 5.7))
    for x, name, second, arm in pos:
        rs = by_arm(results(name))[arm]
        done = [r for r in rs if r["stop_reason"] != PENDING]
        pend = len(rs) - len(done)
        vals = [r["normalized_regret"] for r in done]
        hits = sum(r["hit"] for r in done)
        hatch = second and ARM_TYPE[arm] != "baseline"
        ax.bar(x, float(np.mean(vals)), width=0.72, **fs.bar_kwargs(ARM_TYPE[arm], hatch))
        fs.points(ax, x, sorted(vals), spread=0.26)
        label = f"{hits}/{len(done)}" + (f"\n+{pend}p" if pend else "")
        ax.text(x, 1.36, label, ha="center", va="bottom", fontsize=10)
    ax.text(pos[0][0] - 0.65, 1.36, "hit", ha="right", va="bottom", fontsize=10, color=fs.MUTED)
    ax.set_ylim(0, 1.55)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ax.axhline(1.0, color=fs.MUTED, lw=0.8, ls="--", zorder=1)
    ax.set_ylabel("normalised regret of the declared candidate\n(0 = optimum, 1 = random pick)")
    label_groups(ax, pos)
    ax.set_title(
        "Regret per campaign by arm (bar: mean; points: campaigns; top: optimum hit k/n;\n"
        "+Np: N episodes pending, not run because of the dollar cap).\n" + SAMPLE,
        fontsize=10.5,
        loc="left",
    )
    fig.text(0.01, 0.01, KEY, fontsize=10, color=fs.MUTED, va="bottom")
    fig.subplots_adjust(left=0.1, right=0.99, top=0.76, bottom=0.31)
    fs.save(fig, out / "regret_by_arm.png")


# ---------------------------------------------------------------- figure 2

FRACTIONS = np.linspace(0.0, 1.0, 21)
CURVE_ARMS = [
    ("random_feasible", "random (no model)", fs.BASELINE, ":", "o"),
    ("tuned_acquisition", "tuned UCB (no model)", fs.BASELINE, "-", "s"),
    ("single_call", "single call", fs.SINGLE, "-", "^"),
    ("agent", "agent", fs.AGENT, "-", "D"),
]


def oracles(family: str) -> tuple[dict, dict]:
    cands, _ = load_candidates(ROOT / "tests" / "fixtures" / "candidates.json")
    specs = {
        s.campaign_id: s for s in load_campaigns(ROOT / "evaluator" / f"campaigns_{family}.json")
    }
    return specs, {cid: Simulator(s, cands, 0).oracle_summary(cands) for cid, s in specs.items()}


def anytime_regret(ledger: dict, oracle: dict) -> list[float]:
    """Regret of the incumbent (highest mean observed value) at each budget fraction."""
    span = oracle["f_star"] - oracle["f_mean"]
    spent, steps = 0.0, []
    for rec, obs in zip(ledger["receipts"], ledger["observations"], strict=True):
        spent += rec["charged"]
        steps.append((spent, obs))
    out = []
    for f in FRACTIONS:
        vals: dict[str, list[float]] = defaultdict(list)
        for s, o in steps:
            if s <= f * ledger["budget_units"] + 1e-9 and o["value"] is not None:
                vals[o["candidate_id"]].append(o["value"])
        if not vals:
            out.append(1.0)
            continue
        best = max(sorted(vals), key=lambda c: float(np.mean(vals[c])))
        out.append((oracle["f_star"] - oracle["values"][best]) / span)
    return out


def curves(name: str, family: str, arm: str) -> np.ndarray:
    _, orc = oracles(family)
    rows = [r for r in results(name) if r["arm"] == arm and r["stop_reason"] != PENDING]
    data = []
    for r in rows:
        path = (
            ROOT
            / "reports"
            / name
            / "ledgers"
            / family
            / arm
            / f"{r['campaign_id']}_run{r['run']}.json"
        )
        data.append(
            anytime_regret(json.loads(path.read_text(encoding="utf-8")), orc[r["campaign_id"]])
        )
    return np.array(data)


def regret_vs_spend(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(fs.WIDTH, 4.6), sharey=True)
    panels = [
        (axes[0], "live", "heldout", "v1, gpt-4.1-mini, held-out v1"),
        (axes[1], "live_v2", "heldout_v2", "v2, gpt-4.1-mini, held-out v2"),
    ]
    for ax, name, family, title in panels:
        for arm, _label, colour, ls, marker in CURVE_ARMS:
            c = curves(name, family, arm)
            med = np.median(c, axis=0)
            q1, q3 = np.percentile(c, [25, 75], axis=0)
            ax.plot(
                FRACTIONS,
                med,
                color=colour,
                ls=ls,
                lw=1.8,
                marker=marker,
                ms=4,
                markevery=4,
                markerfacecolor=colour,
                markeredgecolor=fs.EDGE,
                zorder=3,
            )
            if arm in ("tuned_acquisition", "agent"):
                ax.fill_between(FRACTIONS, q1, q3, color=colour, alpha=0.18, lw=0, zorder=2)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("fraction of the assay budget spent")
        ax.set_xlim(0, 1)
    if True:
        c = curves("live_v2_sonnet", "heldout_v2", "agent")
        axes[1].plot(
            FRACTIONS,
            np.median(c, axis=0),
            color=fs.AGENT,
            lw=1.4,
            ls="--",
            marker="D",
            ms=5,
            markevery=4,
            markerfacecolor="white",
            markeredgecolor=fs.EDGE,
            zorder=4,
        )
    axes[0].set_ylim(0, 1.3)
    axes[0].set_ylabel("regret of the incumbent\n(median over campaigns)")
    handles = [
        Line2D(
            [],
            [],
            color=col,
            ls=ls,
            marker=m,
            ms=5,
            markerfacecolor=col,
            markeredgecolor=fs.EDGE,
            label=lab,
        )
        for _, lab, col, ls, m in CURVE_ARMS
    ] + [
        Line2D(
            [],
            [],
            color=fs.AGENT,
            ls="--",
            marker="D",
            ms=5,
            markerfacecolor="white",
            markeredgecolor=fs.EDGE,
            label="agent, second model (v2, 8 campaigns)",
        ),
        Patch(color=fs.BASELINE, alpha=0.3, label="interquartile band (tuned UCB, agent)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(
        "Regret of the best-looking candidate so far, as the budget is spent (no answer yet = 1)\n"
        "Live. gpt-4.1-mini, 10 campaigns x 2 streams per panel; second model 8 campaigns x 1",
        fontsize=10.5,
        x=0.01,
        ha="left",
    )
    fig.subplots_adjust(left=0.1, right=0.97, top=0.82, bottom=0.31, wspace=0.1)
    fs.save(fig, out / "regret_vs_spend.png")


# ---------------------------------------------------------------- figure 3

STOPS = [
    ("finished", "finished", fs.OUTCOME["correct"]),
    ("max_tokens", "token cap (80,000)", fs.OUTCOME["wrong"]),
    ("max_tool_calls", "tool-call cap (36)", fs.OUTCOME["failed"]),
    (PENDING, "pending (dollar cap)", fs.OUTCOME["not run"]),
]
LLM_ARMS = ["single_call", "agent", "agent_no_replicate", "agent_default_selector"]


def stop_reasons(out: Path) -> None:
    pos = layout(LLM_ARMS)
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 5.0))
    for x, name, second, arm in pos:
        rs = by_arm(results(name))[arm]
        n = len(rs)
        bottom = 0.0
        for key, _, colour in STOPS:
            k = sum(r["stop_reason"] == key for r in rs)
            if k:
                ax.bar(
                    x,
                    k / n,
                    bottom=bottom,
                    width=0.72,
                    color=colour,
                    edgecolor=fs.EDGE,
                    linewidth=0.8,
                    hatch=fs.SECOND_MODEL_HATCH if second else None,
                    zorder=2,
                )
                bottom += k / n
        k = sum(r["stop_reason"] == "finished" for r in rs)
        ax.text(x, 1.03, f"{k}/{n}", ha="center", va="bottom", fontsize=10)
    ax.text(
        pos[0][0] - 0.65, 1.03, "finished", ha="right", va="bottom", fontsize=10, color=fs.MUTED
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share of episodes")
    label_groups(ax, pos)
    ax.set_xlim(pos[0][0] - 1.6, pos[-1][0] + 1.0)
    handles = [Patch(facecolor=c, edgecolor=fs.EDGE, label=lab) for _, lab, c in STOPS]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=4, frameon=False
    )
    ax.set_title(
        "How LLM episodes ended (non-LLM arms always finish; aborted HTTP 429 attempt not shown)\n"
        + SAMPLE,
        fontsize=10.5,
        loc="left",
    )
    fig.subplots_adjust(left=0.1, right=0.99, top=0.8, bottom=0.33)
    fs.save(fig, out / "stop_reasons.png")


# ---------------------------------------------------------------- figure 4


def pick_example() -> dict:
    """Lower-median v2 run-0 agent campaign after sorting by regret, then campaign order."""
    specs, _ = oracles("heldout_v2")
    order = list(specs)
    rows = [r for r in results("live_v2") if r["arm"] == "agent" and r["run"] == 0]
    rows.sort(key=lambda r: (r["normalized_regret"], order.index(r["campaign_id"])))
    return rows[(len(rows) - 1) // 2]


def example_campaign(out: Path) -> None:
    row = pick_example()
    cid = row["campaign_id"]
    specs, orc = oracles("heldout_v2")
    trace = ROOT / "reports" / "live_v2" / "traces" / "heldout_v2" / "agent" / f"{cid}_run0.jsonl"
    events = [json.loads(x) for x in trace.read_text(encoding="utf-8").splitlines() if x.strip()]
    steps = [e for e in events if e.get("event") == "step"]
    stop = next(e for e in events if e.get("event") == "stop")
    budget = specs[cid].budget_units
    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(fs.WIDTH, 5.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4]}
    )
    kinds = {
        "screen": ("o", "screen"),
        "confirm": ("D", "confirm"),
        "replicate": ("s", "replicate"),
    }
    spent, xs, ys = 0.0, [0], [0.0]
    for e in steps:
        res = e["result"]
        tool = e["action"]["tool"]
        if tool in ("measure", "replicate") and res["ok"]:
            o = res["output"]
            if not o.get("deduplicated"):
                spent += o["cost_charged"]
            kind = "replicate" if tool == "replicate" else o["fidelity"]
            if o["value"] is not None:
                top.scatter(
                    e["step"],
                    o["value"],
                    marker=kinds[kind][0],
                    s=34,
                    color=fs.AGENT,
                    edgecolor=fs.EDGE,
                    linewidth=0.8,
                    zorder=3,
                )
        elif tool in ("propose_candidates", "check_feasibility", "surrogate_summary"):
            top.plot(
                [e["step"]] * 2,
                [-0.02, 0.02],
                color=fs.MUTED,
                lw=1.0,
                transform=top.get_xaxis_transform(),
            )
        xs.append(e["step"])
        ys.append(spent)
    f_star = orc[cid]["f_star"]
    top.axhline(f_star, color=fs.EDGE, ls="--", lw=1.0)
    top.text(
        steps[-1]["step"] + 0.5,
        f_star,
        "true best feasible value",
        va="bottom",
        ha="right",
        fontsize=10,
    )
    top.set_ylabel("observed synthetic_score")
    bot.step(xs, ys, where="post", color=fs.AGENT, lw=1.8)
    bot.axhline(budget, color=fs.MUTED, ls=":", lw=1.0)
    bot.text(0.5, budget, f"budget {budget:g} units", va="bottom", fontsize=10, color=fs.MUTED)
    bot.axvline(steps[-1]["step"], color=fs.OUTCOME["wrong"], lw=1.4)
    bot.text(
        steps[-1]["step"] - 0.4,
        budget * 1.08,
        f"stop: {stop['stop_reason']}",
        ha="right",
        fontsize=10,
        color=fs.OUTCOME["wrong"],
    )
    bot.set_ylim(0, budget * 1.25)
    bot.set_ylabel("units spent")
    bot.set_xlabel("agent step")
    handles = [
        Line2D([], [], ls="", marker=m, ms=6, color=fs.AGENT, markeredgecolor=fs.EDGE, label=lab)
        for m, lab in kinds.values()
    ] + [Line2D([], [], color=fs.MUTED, lw=1.0, label="propose, check or summary (tick)")]
    top.legend(handles=handles, loc="center left", ncol=1, frameon=False, fontsize=10)
    fig.suptitle(
        f"Example campaign {cid}, v2 run 0, agent (gpt-4.1-mini); "
        f"regret {row['normalized_regret']:.3f}\n"
        "Selection rule: lower median (5th of 10) of the v2 run-0 agent campaigns "
        "sorted by regret,\nthen by campaign order",
        fontsize=10.5,
        x=0.01,
        ha="left",
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.83, bottom=0.1, hspace=0.12)
    fs.save(fig, out / "example_campaign.png")


def compact(path: Path) -> None:
    """Re-encode a saved PNG with a palette of its 256 most frequent colours.

    Kept colours are stored exactly; every other colour (rare anti-aliasing
    shades) is mapped to the nearest kept colour in RGB space.
    """
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    flat = rgb.reshape(-1, 3)
    colours, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    order = np.lexsort((colours[:, 2], colours[:, 1], colours[:, 0], -counts))
    keep = colours[order[:256]]
    dist = ((colours[:, None, :] - keep[None, :, :]) ** 2).sum(-1)
    index = dist.argmin(1).astype(np.uint8)[inverse.ravel()].reshape(rgb.shape[:2])
    img = Image.fromarray(index, mode="P")
    pal = keep.astype(np.uint8).ravel().tolist()
    img.putpalette(pal + [0] * (768 - len(pal)))
    img.save(path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    fs.apply()
    regret_by_arm(args.out)
    regret_vs_spend(args.out)
    stop_reasons(args.out)
    example_campaign(args.out)
    for name in FIGURES:
        p = args.out / name
        compact(p)
        print(f"{p.name}: {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
