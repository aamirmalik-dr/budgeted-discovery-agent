"""Draw the README figures from saved run outputs only.

Reads reports/live (v1, gpt-4.1-mini), reports/live_v2 (v2 run A,
gpt-4.1-mini) and reports/live_v2_sonnet (v2 run B, second model):
task_results.jsonl, the per-campaign receipt ledgers and, for the example
campaign, the saved agent trace. True values for the anytime-regret curves and
the example campaign come from the frozen campaign definitions in evaluator/
through the deterministic simulator, read after the fact as the scorer does.
No model is called and nothing is estimated. The aborted run B attempt
(reports/aborted/) is not drawn.

Each figure is written to docs/figures/ as SVG and as a 300 dpi PNG, plus
figure_stats.json with every plotted value and interval. Because this
repository already carries large saved outputs, each PNG is re-encoded after
saving with a 256-colour palette built from its own most frequent colours: the
hardcoded fills, lines and text keep their exact colours, and only rare
anti-aliasing shades map to the nearest kept colour. Running the script twice
gives byte-identical files.

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
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from budgeted_discovery_agent.candidates import load_candidates  # noqa: E402
from budgeted_discovery_agent.environment import Simulator, load_campaigns  # noqa: E402

# (output dir, campaign family, label, second model)
ROUNDS = [
    ("live", "heldout", "v1", False),
    ("live_v2", "heldout_v2", "v2", False),
    ("live_v2_sonnet", "heldout_v2", "v2, Sonnet", True),
]
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
ARM_NAME = {
    "random_feasible": "Random feasible",
    "cost_aware": "Cost-aware design",
    "tuned_acquisition": "Tuned GP-UCB",
    "single_call": "Single call",
    "agent": "Agent",
    "agent_no_replicate": "Agent, no replicate",
    "agent_default_selector": "Agent, default selector",
}
LLM_ARMS = ["single_call", "agent", "agent_no_replicate", "agent_default_selector"]
PENDING = "run_budget_exhausted"
STEMS = ["fig1_main", "figS1_paired_vs_tuned"]
TUNED_SHADE = "#4D4D4D"  # darker grey and a down-triangle tell the tuned rule from random
FIG_H = 190.0
STATS: dict[str, dict] = {}


def results(name: str) -> list[dict]:
    path = ROOT / "reports" / name / "task_results.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def by_arm(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["arm"]].append(r)
    return out


def done(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["stop_reason"] != PENDING]


def oracles(family: str) -> tuple[dict, dict]:
    cands, _ = load_candidates(ROOT / "tests" / "fixtures" / "candidates.json")
    specs = {
        s.campaign_id: s for s in load_campaigns(ROOT / "evaluator" / f"campaigns_{family}.json")
    }
    return specs, {cid: Simulator(s, cands, 0).oracle_summary(cands) for cid, s in specs.items()}


def per_campaign(rows: list[dict], key: str) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in done(rows):
        acc[r["campaign_id"]].append(float(r[key]))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def paired(rows_a: list[dict], rows_b: list[dict], tag: str) -> tuple[float, float, float, int]:
    a, b = per_campaign(rows_a, "normalized_regret"), per_campaign(rows_b, "normalized_regret")
    camps = sorted(set(a) & set(b))
    m, lo, hi = fs.paired_bootstrap([a[c] for c in camps], [b[c] for c in camps])
    STATS[tag] = {"mean_difference": m, "bootstrap95": [lo, hi], "n_campaigns": len(camps)}
    return m, lo, hi, len(camps)


# Anytime regret --------------------------------------------------------------

FRACTIONS = np.linspace(0.0, 1.0, 21)


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


def curves(name: str, family: str, arm: str, orc: dict) -> tuple[np.ndarray, list[str]]:
    rows = done([r for r in results(name) if r["arm"] == arm])
    data, camps = [], []
    for r in rows:
        fname = f"{r['campaign_id']}_run{r['run']}.json"
        path = ROOT / "reports" / name / "ledgers" / family / arm / fname
        ledger = json.loads(path.read_text(encoding="utf-8"))
        data.append(anytime_regret(ledger, orc[r["campaign_id"]]))
        camps.append(r["campaign_id"])
    return np.array(data), camps


def median_band(c: np.ndarray, camps: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median curve with a bootstrap 95% band that resamples campaigns (all their runs)."""
    ids = sorted(set(camps))
    groups = [np.where(np.asarray(camps) == i)[0] for i in ids]
    rng = np.random.default_rng(fs.SEED)
    meds = np.empty((fs.N_BOOT, c.shape[1]))
    for b in range(fs.N_BOOT):
        pick = rng.integers(0, len(ids), size=len(ids))
        rows = np.concatenate([groups[k] for k in pick])
        meds[b] = np.median(c[rows], axis=0)
    lo, hi = np.percentile(meds, [2.5, 97.5], axis=0)
    return np.median(c, axis=0), lo, hi


CURVE_ARMS = ["random_feasible", "tuned_acquisition", "single_call", "agent"]


def panel_curves(fig, top: float) -> None:
    left0, width, height, gapx, gapy = 22.0, 33.0, 19.0, 5.0, 5.0
    axes, ymax = [], 1.0
    for i, (name, family, label, _) in enumerate(ROUNDS[:2]):
        _, orc = oracles(family)
        cs = {arm: curves(name, family, arm, orc) for arm in CURVE_ARMS}
        meds = {arm: median_band(*cs[arm]) for arm in CURVE_ARMS}
        if i == 1:
            sc = curves("live_v2_sonnet", family, "agent", orc)
            sonnet = median_band(*sc)
            stc = curves("live_v2_sonnet", family, "tuned_acquisition", orc)
            sonnet_tuned = median_band(*stc)
            STATS["incumbent_regret_median/live_v2_sonnet/tuned_acquisition"] = {
                "fractions": FRACTIONS.tolist(),
                "median": sonnet_tuned[0].tolist(),
                "bootstrap95": [sonnet_tuned[1].tolist(), sonnet_tuned[2].tolist()],
                "n_runs": len(stc[1]),
            }
            STATS["incumbent_regret_median/live_v2_sonnet/agent"] = {
                "fractions": FRACTIONS.tolist(),
                "median": sonnet[0].tolist(),
                "bootstrap95": [sonnet[1].tolist(), sonnet[2].tolist()],
                "n_runs": len(sc[1]),
            }
        for j, arm in enumerate(CURVE_ARMS):
            ax = fs.axes_mm(
                fig, left0 + j * (width + gapx), top + i * (height + gapy), width, height
            )
            for other in CURVE_ARMS:
                if other != arm:
                    ax.step(
                        FRACTIONS,
                        meds[other][0],
                        where="post",
                        color=(
                            TUNED_SHADE
                            if other == "tuned_acquisition" and arm != "random_feasible"
                            else fs.CONTEXT
                        ),
                        linewidth=0.8,
                        zorder=1,
                    )
            m, lo, hi = meds[arm]
            STATS[f"incumbent_regret_median/{name}/{arm}"] = {
                "fractions": FRACTIONS.tolist(),
                "median": m.tolist(),
                "bootstrap95": [lo.tolist(), hi.tolist()],
                "n_runs": len(cs[arm][1]),
            }
            t = ARM_TYPE[arm]
            ax.fill_between(
                FRACTIONS,
                lo,
                hi,
                step="post",
                color=fs.ARM_COLOUR[t],
                alpha=0.22,
                linewidth=0,
                zorder=2,
            )
            lk = fs.line_kw(t, width=1.2)
            if arm == "tuned_acquisition":
                lk["color"] = TUNED_SHADE
            ax.step(FRACTIONS, m, where="post", **lk)
            if i == 1 and arm == "agent":
                for edge in (sonnet[1], sonnet[2]):
                    ax.step(
                        FRACTIONS,
                        edge,
                        where="post",
                        color=fs.AGENT,
                        linewidth=0.4,
                        linestyle=(0, (1, 1.2)),
                        zorder=2,
                    )
                ax.step(FRACTIONS, sonnet[0], where="post", **fs.line_kw("agent", True, 1.0))
                ax.step(
                    FRACTIONS,
                    sonnet_tuned[0],
                    where="post",
                    color=TUNED_SHADE,
                    linewidth=0.8,
                    linestyle=(0, (2.2, 1.2)),
                    zorder=2,
                )
                ymax = max(ymax, float(sonnet[2].max()))
            ax.set_xlim(0, 1)
            axes.append(ax)
            ymax = max(ymax, float(hi.max()), float(m.max()))
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            fs.tidy_ticks(ax)
            if j:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_xticklabels([])
                ax.set_title(ARM_NAME[arm])
        fig.text(
            (left0 + 4 * width + 3 * gapx + 3.0) / fs.FULL_WIDTH,
            1 - (top + i * (height + gapy) + height / 2) / FIG_H,
            f"{label}\ngpt-4.1-mini",
            ha="center",
            va="center",
            rotation=270,
        )
    fig.text(
        (left0 + 2 * width + 1.5 * gapx) / fs.FULL_WIDTH,
        1 - (top + 2 * height + gapy + 8) / FIG_H,
        "Budget spent (fraction)",
        ha="center",
        va="center",
    )
    fig.text(
        (left0 - 8) / fs.FULL_WIDTH,
        1 - (top + height + gapy / 2) / FIG_H,
        "Incumbent regret (median)",
        rotation=90,
        ha="center",
        va="center",
    )
    for ax in axes:
        ax.set_ylim(0, ymax * 1.03)
        ax.spines["left"].set_bounds(0, ymax)
    fs.panel_label(fig, "a", 1.5, top - 6.0)


def raincloud_rows() -> list[tuple[float, str, str, bool, str]]:
    rows, y = [], 0.0
    for arm in CURVE_ARMS:
        for name, _, label, second in ROUNDS:
            if arm in by_arm(results(name)):
                rows.append((y, arm, name, second, label))
                y += 2.0
        y += 0.4
    return rows


def panel_final(fig, top: float) -> None:
    rows = raincloud_rows()
    height = (rows[-1][0] + 1.0) * 2.7
    ax = fs.axes_mm(fig, 41.0, top, 44.0, height)
    ax.set_ylim(rows[-1][0] + 0.8, -0.8)
    for y, arm, name, second, _ in rows:
        rs = done(by_arm(results(name))[arm])
        vals = [r["normalized_regret"] for r in rs]
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        STATS[f"final_regret/{name}/{arm}"] = {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(med),
            "q1": float(q1),
            "q3": float(q3),
            "max": float(max(vals)),
            "pending": len(by_arm(results(name))[arm]) - len(rs),
        }
        tuned = arm == "tuned_acquisition"
        fs.raincloud(
            ax,
            y,
            vals,
            ARM_TYPE[arm],
            second,
            height=0.36,
            jitter=0.62,
            marker="v" if tuned else None,
            shade=TUNED_SHADE if tuned else None,
        )
        none = sum(1 for r in rs if r["chosen"] is None)
        STATS[f"final_regret/{name}/{arm}"]["no_answer"] = none
        if none:
            ax.text(
                1.02,
                y,
                f"{none} no answer",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                color=fs.MUTED,
            )
    ax.axvline(1.0, color=fs.REFERENCE, linewidth=0.6, zorder=0)
    ax.set_yticks([r[0] for r in rows], [r[4] for r in rows])
    for arm in CURVE_ARMS:
        ys = [r[0] for r in rows if r[1] == arm]
        ax.text(
            -0.3,
            (ys[0] + ys[-1]) / 2,
            ARM_NAME[arm].replace(", ", ",\n"),
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
        )
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    xmax = max(STATS[k]["max"] for k in STATS if k.startswith("final_regret/"))
    ax.set_xlim(-0.03, max(1.12, xmax + 0.04))
    ax.spines["bottom"].set_bounds(0, max(1.1, xmax))
    ax.set_xticks([0, 0.5, 1])
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    fs.tidy_ticks(ax, "x")
    ax.set_xlabel("Final normalised regret")
    fs.panel_label(fig, "b", 1.5, top - 3.0)


STOPS = [
    ("finished", "correct", "Finished"),
    ("max_tokens", "wrong", "Token cap (80,000)"),
    ("max_tool_calls", "failed", "Tool-call cap (36)"),
    (PENDING, "pending", "Pending (dollar cap)"),
]


def panel_stops(fig, top: float) -> None:
    rows, y = [], 0.0
    for arm in LLM_ARMS:
        for name, _, label, second in ROUNDS:
            if arm in by_arm(results(name)):
                rows.append((y, arm, name, second, label))
                y += 1
        y += 0.5
    height = (rows[-1][0] + 1) * 3.1
    ax = fs.axes_mm(fig, 138.0, top, 31.0, height)
    for y, arm, name, second, _ in rows:
        rs = by_arm(results(name))[arm]
        n = len(rs)
        start = 0.0
        counts = {}
        for key, colour, _ in STOPS:
            k = sum(r["stop_reason"] == key for r in rs)
            counts[key] = k
            if k:
                ax.barh(
                    y,
                    k / n,
                    left=start,
                    height=0.66,
                    color=fs.OUTCOME[colour],
                    edgecolor=fs.MUTED if colour == "pending" else "white",
                    linewidth=0.4,
                )
            start += k / n
        if second:
            ax.add_patch(
                fs.plt.Rectangle(
                    (0, y - 0.33),
                    1,
                    0.66,
                    fill=False,
                    edgecolor=fs.INK,
                    linewidth=0.75,
                    linestyle=(0, (2.2, 1.2)),
                    zorder=3,
                )
            )
        STATS[f"stop_reasons/{name}/{arm}"] = counts | {"n": n}
        ax.text(
            1.03,
            y,
            f"{counts['finished']}/{n}",
            va="center",
            ha="left",
            transform=ax.get_yaxis_transform(),
            color=fs.MUTED,
        )
    ax.set_ylim(rows[-1][0] + 0.6, -0.6)
    ax.set_yticks([r[0] for r in rows], [r[4] for r in rows])
    for arm in LLM_ARMS:
        ys = [r[0] for r in rows if r[1] == arm]
        ax.text(
            -0.5,
            (ys[0] + ys[-1]) / 2,
            ARM_NAME[arm].replace(", ", ",\n"),
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
        )
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    fs.tidy_ticks(ax, "x")
    ax.set_xlabel("Episodes (fraction)")
    ax.text(
        1.03,
        -1.0,
        "Finished",
        va="center",
        ha="left",
        transform=ax.get_yaxis_transform(),
        color=fs.MUTED,
    )
    key = fs.axes_mm(fig, 100.0, top + height + 10.0, 82.0, 7.0)
    key.set_axis_off()
    key.legend(
        handles=fs.outcome_handles([(c, lab) for _, c, lab in STOPS]),
        loc="center",
        ncol=2,
        handlelength=1.0,
        columnspacing=1.0,
    )
    fs.panel_label(fig, "c", 100.0, top - 3.0)


def pick_example() -> dict:
    """Lower-median v2 run-0 agent campaign after sorting by regret, then campaign order."""
    specs, _ = oracles("heldout_v2")
    order = list(specs)
    rows = [r for r in results("live_v2") if r["arm"] == "agent" and r["run"] == 0]
    rows.sort(key=lambda r: (r["normalized_regret"], order.index(r["campaign_id"])))
    return rows[(len(rows) - 1) // 2]


EVENTS = [
    ("screen", "Screen assay"),
    ("confirm", "Confirm assay"),
    ("replicate", "Replicate"),
    ("propose_candidates", "Propose"),
    ("check_feasibility", "Check feasibility"),
    ("surrogate_summary", "Surrogate summary"),
]


def panel_example(fig, top: float) -> None:
    row = pick_example()
    cid = row["campaign_id"]
    specs, orc = oracles("heldout_v2")
    oracle = orc[cid]
    span = oracle["f_star"] - oracle["f_mean"]
    trace = ROOT / "reports" / "live_v2" / "traces" / "heldout_v2" / "agent" / f"{cid}_run0.jsonl"
    events = [json.loads(x) for x in trace.read_text(encoding="utf-8").splitlines() if x.strip()]
    steps = [e for e in events if e.get("event") == "step"]
    stop = next(e for e in events if e.get("event") == "stop")
    left, width = 41.0, 122.0
    ra = fs.axes_mm(fig, left, top, width, 17.0)
    rc = fs.axes_mm(fig, left, top + 19.5, width, 20.0)
    obs: dict[str, list[float]] = defaultdict(list)
    xs, ys, spent = [0], [1.0], 0.0
    kinds = [k for k, _ in EVENTS]
    for e in steps:
        tool = e["action"]["tool"]
        res = e["result"]
        kind = tool
        if tool in ("measure", "replicate") and res["ok"]:
            o = res["output"]
            kind = "replicate" if tool == "replicate" else o["fidelity"]
            if not o.get("deduplicated"):
                spent += o["cost_charged"]
            if o["value"] is not None:
                obs[o["candidate_id"]].append(o["value"])
        if kind in kinds:
            yk = kinds.index(kind)
            assay = kind in ("screen", "confirm", "replicate")
            ra.plot(
                [e["step"]],
                [yk],
                marker="|",
                markersize=6,
                markeredgewidth=1.4,
                color=fs.AGENT if assay else fs.INK,
                linestyle="none",
            )
        if obs:
            best = max(sorted(obs), key=lambda c: float(np.mean(obs[c])))
            ys.append((oracle["f_star"] - oracle["values"][best]) / span)
        else:
            ys.append(1.0)
        xs.append(e["step"])
    STATS[f"example_campaign/{cid}"] = {
        "selection": "lower median (5th of 10) of the v2 run-0 agent campaigns by regret",
        "declared_regret": row["normalized_regret"],
        "steps": len(steps),
        "assays": sum(1 for e in steps if e["action"]["tool"] in ("measure", "replicate")),
        "units_spent": spent,
        "budget_units": specs[cid].budget_units,
        "stop_reason": stop["stop_reason"],
        "final_incumbent_regret": ys[-1],
    }
    last = steps[-1]["step"]
    ra.set_ylim(len(EVENTS) - 0.5, -0.5)
    ra.set_yticks(range(len(EVENTS)), [lab for _, lab in EVENTS])
    ra.tick_params(axis="y", length=0)
    ra.spines["left"].set_visible(False)
    ra.spines["bottom"].set_visible(False)
    ra.tick_params(axis="x", length=0, labelbottom=False)
    for k in range(len(EVENTS)):
        ra.axhline(k, color=fs.REFERENCE, linewidth=0.4, zorder=0)
    rc.step(xs, ys, where="post", **fs.line_kw("agent", width=1.2))
    rc.axhline(row["normalized_regret"], color=fs.MUTED, linewidth=0.6, linestyle=(0, (1, 1.5)))
    rc.text(
        9.0,
        0.93,
        "declared answer: none (token cap), scored as regret 1",
        ha="left",
        va="top",
        color=fs.MUTED,
    )
    for a in (ra, rc):
        a.set_xlim(0, last + 1)
        a.axvline(last + 0.5, color=fs.INK, linewidth=0.7, zorder=0)
    rc.text(last + 0.8, 0.8, "stop:\ntoken cap", ha="left", va="center", color=fs.INK)
    rc.text(
        last + 0.8,
        0.25,
        f"{spent:g} of {specs[cid].budget_units:g}\nunits spent",
        ha="left",
        va="center",
        color=fs.MUTED,
    )
    rc.set_ylim(0, max(1.08, max(ys) * 1.05))
    rc.set_yticks([0, 0.5, 1])
    fs.tidy_ticks(rc, "y")
    rc.xaxis.set_major_locator(MultipleLocator(5))
    rc.xaxis.set_minor_locator(MultipleLocator(1))
    rc.set_xlabel("Agent step")
    rc.set_ylabel("Incumbent\nregret")
    fs.panel_label(fig, "d", 1.5, top - 3.0)


def figure1(out: Path) -> list[Path]:
    fig = fs.figure(fs.FULL_WIDTH, FIG_H)
    panel_curves(fig, top=9.0)
    panel_final(fig, top=71.0)
    panel_stops(fig, top=71.0)
    panel_example(fig, top=140.0)
    return save(fig, out, "fig1_main")


# Supplementary --------------------------------------------------------------


def figure_s1(out: Path) -> list[Path]:
    rows = []
    for arm in LLM_ARMS:
        for name, _, label, second in ROUNDS:
            data = by_arm(results(name))
            if arm in data:
                m, lo, hi, n = paired(
                    data[arm], data["tuned_acquisition"], f"paired_vs_tuned/{name}/{arm}"
                )
                rows.append((f"{ARM_NAME[arm]}, {label}", arm, second, m, lo, hi, n))
    v2 = by_arm(results("live_v2"))
    paired(v2["agent"], v2["single_call"], "paired_vs_single_call/live_v2/agent")
    ys, y = [], 0.0
    for r in rows:
        y += 0.5 if r[2] else 0.0
        ys.append(y)
        y += 1.5 if r[2] else 1.0
    fig = fs.figure(fs.SINGLE_WIDTH * 1.35, 64)
    ax = fs.axes_mm(fig, 46, 4, 62, 46)
    for i, (_, arm, second, m, lo, hi, _n) in zip(ys, rows, strict=True):
        fs.dot_interval(ax, i, m, lo, hi, ARM_TYPE[arm], second)
        ax.text(
            1.02,
            i,
            f"n = {_n}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            color=fs.MUTED,
        )
    fs.ref_line(ax, 0.0, "x")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_ylim(ys[-1] + 0.5, -1.3)
    ax.set_xlim(-0.45, 1.02)
    ax.spines["bottom"].set_bounds(-0.4, 1.0)
    ax.set_xticks([-0.4, 0, 0.4, 0.8, 1.0])
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    fs.tidy_ticks(ax, "x")
    ax.set_xlabel("Paired difference in final regret\nvs tuned GP-UCB (arm minus rule)")
    ax.text(-0.02, -1.0, "arm better", ha="right", va="center", color=fs.MUTED)
    ax.text(0.02, -1.0, "tuned rule better", ha="left", va="center", color=fs.MUTED)
    return save(fig, out, "figS1_paired_vs_tuned")


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
    img.save(path, optimize=True, dpi=(fs.DPI, fs.DPI))


def save(fig, out: Path, stem: str) -> list[Path]:
    paths = fs.save(fig, out, stem)
    compact(out / f"{stem}.png")
    return paths


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args(argv)
    fs.apply()
    paths = figure1(args.out)
    paths += figure_s1(args.out)

    def rnd(v):
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, list):
            return [rnd(x) for x in v]
        if isinstance(v, dict):
            return {k: rnd(x) for k, x in v.items()}
        return v

    meta = {
        "interval": "percentile bootstrap 95% over campaigns for median curves and paired "
        "differences",
        "bootstrap_resamples": fs.N_BOOT,
        "seed": fs.SEED,
        "resampling_unit": "campaign",
    }
    stats = args.out / "figure_stats.json"
    text = json.dumps({"method": meta, "values": rnd(dict(sorted(STATS.items())))}, indent=1)
    stats.write_text(text + "\n", encoding="utf-8", newline="\n")
    return paths + [stats]


if __name__ == "__main__":
    for p in main():
        print(f"wrote {p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p}")
