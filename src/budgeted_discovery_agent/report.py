"""Build reports/evidence.json, reports/evidence.md and the README tables.

Everything is read from saved outputs (run_meta.json, comparison.csv,
paired.json, task_results.jsonl). A mode without saved outputs is "pending".
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

MODES = ("fixture", "live")
NON_LLM = ("random_feasible", "cost_aware", "tuned_acquisition")
MARKERS = {
    "measured": ("<!-- measured-table:start -->", "<!-- measured-table:end -->"),
    "scorer": ("<!-- scorer-table:start -->", "<!-- scorer-table:end -->"),
}
NOTES = {
    "fixture": (
        "Fixture mode replays scripted replies for single_call and the three agent arms, with "
        "planted failures. Those rows check the ledger, policy and scorer; they are not model "
        "results. The non-LLM baselines call no model, so their fixture rows are real "
        "measurements on one noise stream per campaign."
    ),
    "live": "Live model results on the held-out family; non-LLM rows share the same noise streams.",
}
LIMITATIONS = [
    "Synthetic response surfaces on a 66-candidate fictional simplex; no real materials data "
    "and no discovery claim.",
    "The surrogate is a fixed-hyperparameter GP and the acquisition rule is textbook UCB; this "
    "is not a new optimisation algorithm.",
    "Normalised regret uses the mean feasible value as the zero-skill reference; a campaign "
    "with no answer or an infeasible answer scores 1.0.",
    "Fixture results for LLM arms are scripted and say nothing about model quality.",
]


def _num(v: str) -> Any:
    if v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in v else f


def load_mode(reports: Path, mode: str) -> dict[str, Any] | None:
    """Saved outputs for one mode, or None."""
    d = reports / mode
    if not (d / "run_meta.json").exists():
        return None
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    with (d / "comparison.csv").open(encoding="utf-8") as fh:
        comparison = [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(fh)]
    rows = [
        json.loads(x)
        for x in (d / "task_results.jsonl").read_text(encoding="utf-8").splitlines()
        if x
    ]
    paired = json.loads((d / "paired.json").read_text(encoding="utf-8"))
    notable = [
        {k: r.get(k) for k in ("family", "arm", "campaign_id", "run", "status", "stop_reason",
                               "chosen", "true_best", "normalized_regret", "infeasible_attempts",
                               "refused_over_budget", "malformed_actions")}
        for r in rows
        if r["status"] != "complete" or r["infeasible_attempts"] or r["refused_over_budget"]
        or r.get("malformed_actions") or r["budget_violation"] or not r["chosen_feasible"]
    ]  # fmt: skip
    return {
        "meta": meta,
        "metrics": comparison,
        "paired": paired,
        "notable": notable,
        "n": len(rows),
    }


def build_evidence(root: Path) -> dict[str, Any]:
    """Assemble the evidence record from saved outputs."""
    ev: dict[str, Any] = {"schema": "evidence-v1", "modes": {}}
    for mode in MODES:
        data = load_mode(root / "reports", mode)
        if data is None:
            ev["modes"][mode] = "pending"
            continue
        m = data["meta"]
        ev["modes"][mode] = {
            "mode": mode,
            "interpretation": NOTES[mode],
            **{k: m[k] for k in ("git_commit", "git_dirty", "model", "prompt_version",
                                 "prompt_sha256",
                                 "config_sha256", "input_hashes", "hardware", "started_at",
                                 "finished_at", "tuned_params", "agentcore_version")},
            "caps": {"episode": m["config"]["episode"], "run": m["config"]["run_caps"]},
            "cost": {k: m.get(k) for k in ("tokens", "usd", "usd_reported_by_provider",
                                           "adapter_retries")},
            "metrics": data["metrics"],
            "paired_vs_tuned_acquisition": data["paired"],
            "notable_episodes": data["notable"],
            "episodes": data["n"],
        }  # fmt: skip
    live_done = ev["modes"]["live"] != "pending"
    ev["limitations"] = LIMITATIONS + [
        (
            "The live pilot is one model, 10 held-out campaigns and two runs at temperature 0: an "
            "exploratory sample with min and max over runs, not a ranking."
            if live_done
            else "No live model has been evaluated yet; live results are pending."
        )
    ]
    ev["reproduce"] = {
        "fixture": "python -m budgeted_discovery_agent evaluate "
        "--config configs/campaign_fixture.yaml",
        "live": "python -m budgeted_discovery_agent evaluate --config configs/campaign_live.yaml "
        "--require-live",
        "report": "python -m budgeted_discovery_agent report",
        "audit": "python -m budgeted_discovery_agent audit",
    }
    return ev


def _frac(r: dict[str, Any], name: str) -> str:
    return f"{r[f'{name}_k']}/{r[f'{name}_n']}"


def _row(label: str, r: dict[str, Any], paired: dict[tuple[str, str], dict]) -> str:
    p = paired.get((r["family"], r["arm"]))
    vs = f"{p['better']} / {p['tie']} / {p['worse']}" if p else "reference"
    toks = r.get("tokens_total")
    usd = r.get("reported_cost_usd_total")
    return (
        f"| {label} | {r['family']} | {r['arm']} | {r['episodes']} "
        f"| {r['regret_mean']:.3f} ({r['regret_run_min']:.3f} to {r['regret_run_max']:.3f}) "
        f"| {_frac(r, 'hit')} | {_frac(r, 'chosen_feasible')} | {_frac(r, 'confirmed_choice')} "
        f"| {vs} | {r['infeasible_attempts_total']} | {r['budget_violation_total']} "
        f"| {r['replications_total']} | {r['unused_units_mean']:.1f} | {r['failed_total']} "
        f"| {toks if toks is not None else 0} | {f'{usd:.4f}' if usd is not None else '0'} |"
    )


HEAD = (
    "| Mode | Family | Arm | Campaigns | Normalised regret, mean (run min to max) | Hit optimum "
    "| Feasible answer | Answer confirmed | vs tuned: better / tie / worse | Infeasible assays "
    "| Budget violations | Replications | Unused units (mean) | Failed | Tokens "
    "| USD (provider) |\n" + "|---" * 16 + "|\n"
)


def _paired_index(data: dict[str, Any]) -> dict[tuple[str, str], dict]:
    return {(p["family"], p["arm"]): p for p in data["paired_vs_tuned_acquisition"]}


def measured_table(ev: dict[str, Any]) -> str:
    """Headline: non-LLM baselines (fixture noise stream) and every live row."""
    lines = []
    fx = ev["modes"]["fixture"]
    if fx != "pending":
        idx = _paired_index(fx)
        lines += [
            _row("no model", r, idx)
            for r in fx["metrics"]
            if r["arm"] in NON_LLM and r["family"] == "dev"
        ]
    lv = ev["modes"]["live"]
    if lv == "pending":
        lines.append("| live: pending |" + " |" * 15)
    else:
        idx = _paired_index(lv)
        lines += [
            _row("no model" if r["arm"] in NON_LLM else "live", r, idx) for r in lv["metrics"]
        ]
    return HEAD + "\n".join(lines) + "\n"


def scorer_table(ev: dict[str, Any]) -> str:
    """Scripted fixture rows for the LLM arms: checks, not model results."""
    fx = ev["modes"]["fixture"]
    if fx == "pending":
        return HEAD + "| fixture: pending |" + " |" * 15 + "\n"
    idx = _paired_index(fx)
    return (
        HEAD
        + "\n".join(
            _row("fixture, scripted", r, idx) for r in fx["metrics"] if r["arm"] not in NON_LLM
        )
        + "\n"
    )


def evidence_markdown(ev: dict[str, Any]) -> str:
    """Human-readable evidence report."""
    out = ["# Evidence", "", "Generated from saved outputs by "
           "`python -m budgeted_discovery_agent report`. Do not edit by hand.", ""]  # fmt: skip
    for mode in MODES:
        d = ev["modes"][mode]
        out += [f"## {mode.capitalize()} mode", ""]
        if d == "pending":
            out += ["Pending. No saved outputs exist for this mode.", ""]
            continue
        c = d["cost"]
        out += [
            d["interpretation"], "",
            f"- Commit: `{d['git_commit']}` (working tree dirty: {d['git_dirty']})",
            f"- Model: {d['model']}; agentcore {d['agentcore_version']}",
            f"- Prompt: {d['prompt_version']}, sha256 `{d['prompt_sha256'][:16]}`",
            f"- Config sha256: `{d['config_sha256'][:16]}`; tuned acquisition: {d['tuned_params']}",
            f"- Run: {d['started_at']} to {d['finished_at']}",
            f"- Hardware: {d['hardware']['platform']}, Python {d['hardware']['python']}",
            f"- Campaign episodes scored: {d['episodes']}",
            f"- Tokens: {c['tokens']}, USD at listed prices: {c['usd']}, USD reported by provider: "
            f"{c['usd_reported_by_provider']}, adapter retries: {c['adapter_retries']}",
            "", "### Input hashes", "",
        ]  # fmt: skip
        out += [f"- {k}: `{v[:16]}`" for k, v in sorted(d["input_hashes"].items())]
        out += ["", "### Every failed, infeasible, refused or malformed episode", "",
                "| Family | Arm | Campaign | Run | Status | Stop | Chosen | True best | Regret "
                "| Infeasible | Refused | Malformed |", "|---" * 12 + "|"]  # fmt: skip
        out += [
            f"| {n['family']} | {n['arm']} | {n['campaign_id']} | {n['run']} | {n['status']} "
            f"| {n['stop_reason']} | {n['chosen']} | {n['true_best']} | {n['normalized_regret']} "
            f"| {n['infeasible_attempts']} | {n['refused_over_budget']} "
            f"| {n['malformed_actions']} |"
            for n in d["notable_episodes"]
        ]
        out.append("")
    out += ["## Measured results", "", measured_table(ev),
            "## Scorer checks (scripted replies, not model results)", "", scorer_table(ev),
            "## Limitations", ""]  # fmt: skip
    out += [f"- {x}" for x in ev["limitations"]]
    out += ["", "## Reproduce", ""] + [f"- {k}: `{v}`" for k, v in ev["reproduce"].items()]
    return "\n".join(out) + "\n"


def write_reports(root: Path, update_readme: bool = True) -> dict[str, Any]:
    """Write evidence.json and evidence.md; refresh the README tables if markers exist."""
    ev = build_evidence(root)
    rep = root / "reports"
    (rep / "evidence.json").write_text(
        json.dumps(ev, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (rep / "evidence.md").write_text(evidence_markdown(ev), encoding="utf-8", newline="\n")
    readme = root / "README.md"
    if update_readme and readme.exists():
        text = readme.read_text(encoding="utf-8")
        for key, table in (("measured", measured_table(ev)), ("scorer", scorer_table(ev))):
            a, b = MARKERS[key]
            pat = re.compile(re.escape(a) + ".*?" + re.escape(b), re.S)
            if pat.search(text):
                text = pat.sub(lambda _, blk=f"{a}\n{table}{b}": blk, text)
        readme.write_text(text, encoding="utf-8", newline="\n")
    return ev
