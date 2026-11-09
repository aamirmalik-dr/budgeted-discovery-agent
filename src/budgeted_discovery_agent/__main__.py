"""Command line: demo, evaluate, report and audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .accounting import verify
from .agent import EpisodeResult, EpisodeSettings, run_agent
from .agentcore.adapters import FixtureAdapter, LiveModelUnavailable, OpenAICompatibleAdapter
from .candidates import load_candidates
from .environment import load_campaigns
from .evaluation import load_config, run_evaluation
from .report import write_reports
from .schemas import CampaignState

ROOT = Path.cwd()
DEMO_CAMPAIGN = "dev-neartie"


def _short(obj: object, n: int = 60) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _outcome(h) -> str:  # noqa: ANN001
    if not h.result.ok:
        return f"REFUSED {h.result.error.code}: {_short(h.result.error.message, 50)}"
    out = h.result.output or {}
    if "candidates" in out:
        return f"{out['strategy_used']}: " + ", ".join(c["candidate_id"] for c in out["candidates"])
    if "observation_id" in out:
        val = "FAILED" if out["value"] is None else f"{out['value']:.3f}"
        dedup = " (dedup)" if out["deduplicated"] else ""
        return f"{out['observation_id']} {out['fidelity']} {val}, -{out['cost_charged']}{dedup}"
    if "results" in out:
        return ", ".join(f"{r['candidate_id']} {r['status']}" for r in out["results"])
    if "remaining_units" in out:
        return f"remaining {out['remaining_units']}"
    if "top_by_posterior_mean" in out:
        return "top: " + ", ".join(t["candidate_id"] for t in out["top_by_posterior_mean"])
    return _short(out, 50)


def steps_table(state: CampaignState) -> Table:
    """One row per agent step: tool, arguments, outcome, model summary."""
    t = Table(title=f"{state.campaign_id}: decisions")
    for col in ("#", "tool", "arguments", "outcome", "model summary"):
        t.add_column(col, overflow="fold")
    for h in state.history:
        args = {k: v for k, v in h.action.arguments.items() if k != "justification"}
        t.add_row(str(h.step), h.action.tool, _short(args), _outcome(h), _short(h.summary, 50))
    return t


def receipts_table(state: CampaignState) -> Table:
    """The receipt ledger with a running balance."""
    t = Table(title="Receipt ledger")
    for col in (
        "receipt",
        "key",
        "candidate",
        "fidelity",
        "status",
        "reserved",
        "charged",
        "refunded",
        "balance",
    ):
        t.add_column(col)
    bal = state.budget_units
    for r in state.receipts:
        bal = round(bal - r.charged, 9)
        t.add_row(
            r.receipt_id, r.key, r.candidate_id, r.fidelity, r.status,
            f"{r.reserved:g}", f"{r.charged:g}", f"{r.refunded:g}", f"{bal:g}",
        )  # fmt: skip
    return t


def run_demo(
    mode: str, root: Path, console: Console | None = None, model: str | None = None
) -> EpisodeResult:
    """One campaign with the agent; prints decisions, the ledger and its audit."""
    cands, mhash = load_candidates(root / "tests/fixtures/candidates.json")
    spec = next(
        s
        for s in load_campaigns(root / "evaluator/campaigns_dev.json")
        if s.campaign_id == DEMO_CAMPAIGN
    )
    adapter = (
        OpenAICompatibleAdapter.from_env(model=model)
        if mode == "live"
        else FixtureAdapter.from_file(root / "tests/fixtures/scripts/dev_agent.json", DEMO_CAMPAIGN)
    )
    res = run_agent(spec, cands, mhash, adapter, "agent", 0, EpisodeSettings(record_timing=False))
    if console:
        st = res.state
        console.print(
            f"[bold]{spec.campaign_id}[/bold]: budget {st.budget_units:g} units, screen "
            f"{spec.cost['screen']:g}, confirm {spec.cost['confirm']:g}, failed assay charges 50%"
        )
        console.print(steps_table(st))
        console.print(receipts_table(st))
        total = sum(r.charged for r in st.receipts)
        problems = verify(st)
        console.print(
            f"Declared: {st.chosen}. Receipts sum to {total:g} = spent {st.spent_units:g}; "
            f"remaining {st.remaining_units:g}; ledger audit: "
            + ("consistent" if not problems else "; ".join(problems))
        )
        label = "scripted fixture adapter, no model" if mode == "fixture" else adapter.name
        console.print(f"Mode: {mode} ({label}).")
    return res


def audit_ledgers(directory: Path) -> tuple[int, list[str]]:
    """Re-add every saved ledger: receipts must sum to spent, within budget."""
    problems: list[str] = []
    files = sorted(directory.rglob("*.json"))
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        total = round(sum(r["charged"] for r in d["receipts"]), 9)
        if abs(total - d["spent_units"]) > 1e-9:
            problems.append(f"{f.name}: receipts {total} != spent {d['spent_units']}")
        if d["spent_units"] > d["budget_units"] + 1e-9:
            problems.append(
                f"{f.name}: spent {d['spent_units']} exceeds budget {d['budget_units']}"
            )
        keys = [r["key"] for r in d["receipts"]]
        if len(keys) != len(set(keys)):
            problems.append(f"{f.name}: duplicate idempotency key charged")
    return len(files), problems


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m budgeted_discovery_agent``."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="budgeted_discovery_agent", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo", help="run the agent on one dev campaign and print the ledger")
    p.add_argument("--mode", choices=("fixture", "live"), default="fixture")
    p.add_argument("--model", help="model id for live mode; overrides AGENT_MODEL_NAME")
    p = sub.add_parser("evaluate", help="run every arm on every campaign of a config")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--require-live", action="store_true", help="fail unless a live model is configured"
    )
    sub.add_parser("report", help="regenerate evidence files and README tables from saved outputs")
    p = sub.add_parser("audit", help="re-add every saved receipt ledger")
    p.add_argument("--dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    console = Console(width=150)
    try:
        if args.cmd == "demo":
            run_demo(args.mode, ROOT, console, model=args.model)
        elif args.cmd == "evaluate":
            cfg, sha = load_config(args.config)
            if args.require_live and cfg.mode != "live":
                raise LiveModelUnavailable("--require-live needs a config with mode: live")
            out = run_evaluation(cfg, sha, ROOT, require_live=args.require_live)
            console.print(f"wrote {out.relative_to(ROOT).as_posix()}/")
        elif args.cmd == "report":
            ev = write_reports(ROOT)
            console.print(
                "wrote reports/evidence.json, reports/evidence.md and README tables; modes: "
                + ", ".join(
                    f"{k}={'pending' if v == 'pending' else 'saved'}"
                    for k, v in ev["modes"].items()
                )
            )
        elif args.cmd == "audit":
            n, problems = audit_ledgers(args.dir)
            if problems:
                console.print("\n".join(problems))
                return 1
            console.print(f"{n} ledgers audited: every receipt accounted for, no budget exceeded")
    except LiveModelUnavailable as exc:
        console.print(f"[red]live model unavailable:[/red] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
