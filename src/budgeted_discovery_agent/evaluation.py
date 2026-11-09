"""Evaluation: run every arm on every campaign, score against the hidden oracle.

The oracle (true values and the feasible optimum) is consulted here only,
after each campaign has committed its answer. Scoring rules are fixed in
docs/evaluation.md and were written before any evaluation run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field

from . import __version__
from .accounting import verify
from .agent import (
    AGENT_ARMS,
    EpisodeResult,
    EpisodeSettings,
    make_campaign,
    registry_for,
    run_agent,
)
from .agentcore import AGENTCORE_VERSION
from .agentcore.adapters import (
    FixtureAdapter,
    LiveModelUnavailable,
    ModelAdapter,
    OpenAICompatibleAdapter,
)
from .baselines import TunedParams, run_cost_aware, run_random, run_single_call, run_tuned
from .candidates import load_candidates
from .environment import CampaignSpec, Simulator, load_campaigns
from .prompts import PROMPT_VERSION, prompt_hash
from .schemas import Candidate

NON_LLM = ("random_feasible", "cost_aware", "tuned_acquisition")
ALL_ARMS = (*NON_LLM, "single_call", *AGENT_ARMS)
TIE = 0.01
PENDING_STOP = "run_budget_exhausted"
"""Episodes not run because a run cap was reached: reported as pending, not scored."""


class RunCaps(BaseModel):
    max_total_tokens: int | None = None
    max_usd: float | None = None
    usd_per_mtok_input: float | None = None
    usd_per_mtok_output: float | None = None


class EvalConfig(BaseModel):
    """Frozen evaluation settings, loaded from configs/*.yaml."""

    mode: Literal["fixture", "live"]
    model: str | None = None
    families: list[str]
    arms: list[str]
    runs: int = 1
    candidates: str = "tests/fixtures/candidates.json"
    fixture_scripts: str = "tests/fixtures/scripts"
    tuned_params: str = "configs/tuned_acquisition.yaml"
    episode: EpisodeSettings = Field(default_factory=EpisodeSettings)
    run_caps: RunCaps = Field(default_factory=RunCaps)
    output_dir: str = "reports/fixture"
    checkpoint_dir: str = "runs"
    adapter_max_retries: int = 2
    adapter_backoff_s: float = 2.0


def load_config(path: Path) -> tuple[EvalConfig, str]:
    """Load a YAML config; return it with the SHA-256 of the file."""
    raw = Path(path).read_bytes()
    cfg = EvalConfig.model_validate(yaml.safe_load(raw))
    unknown = set(cfg.arms) - set(ALL_ARMS)
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}")
    return cfg, hashlib.sha256(raw).hexdigest()


def load_tuned(path: Path) -> TunedParams:
    """Tuned acquisition parameters (frozen, dev family only)."""
    return TunedParams.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def score(spec: CampaignSpec, cands: list[Candidate], result: EpisodeResult) -> dict[str, Any]:
    """Score one finished campaign against the hidden oracle. See docs/evaluation.md."""
    sim = Simulator(spec, cands, 0)
    oracle = sim.oracle_summary(cands)
    st = result.state
    chosen = st.chosen if result.status == "complete" else None
    feasible = chosen is not None and chosen in oracle["values"]
    span = oracle["f_star"] - oracle["f_mean"]
    regret = (oracle["f_star"] - oracle["values"][chosen]) / span if feasible else 1.0
    ok = [o for o in st.observations if o.status == "ok"]
    problems = verify(st)
    return {
        "chosen": chosen,
        "true_best": oracle["best_id"],
        "normalized_regret": round(max(regret, 0.0), 4),
        "hit": int(chosen == oracle["best_id"]),
        "chosen_feasible": int(feasible),
        "confirmed_choice": int(
            chosen is not None
            and any(o.candidate_id == chosen and o.fidelity == "confirm" for o in ok)
        ),
        "screens": sum(o.fidelity == "screen" for o in st.observations),
        "confirms": sum(o.fidelity == "confirm" for o in st.observations),
        "replications": sum(o.replicate_of is not None for o in st.observations),
        "infeasible_attempts": st.infeasible_attempts,
        "refused_over_budget": st.refused_over_budget,
        "spent_units": st.spent_units,
        "unused_units": st.remaining_units,
        "budget_violation": int(st.spent_units > st.budget_units + 1e-9 or bool(problems)),
        "ledger_problems": problems,
        "failed": int(result.status in ("failed", "stopped")),
        "tool_calls": sum(1 for h in st.history if h.result.ok or h.result.error is not None),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def cost_usd(tok_in: int, tok_out: int, caps: RunCaps) -> float | None:
    """Dollar cost at the configured listed prices, or None if unset."""
    if caps.usd_per_mtok_input is None or caps.usd_per_mtok_output is None:
        return None
    return (tok_in * caps.usd_per_mtok_input + tok_out * caps.usd_per_mtok_output) / 1e6


def run_evaluation(cfg: EvalConfig, cfg_sha: str, root: Path, require_live: bool = False) -> Path:
    """Run every configured arm on every campaign and write the outputs.

    Raises:
        LiveModelUnavailable: Live mode or ``require_live`` without an endpoint.
    """
    live: ModelAdapter | None = None
    if cfg.mode == "live":
        live = OpenAICompatibleAdapter.from_env(
            model=cfg.model,
            max_retries=cfg.adapter_max_retries,
            backoff_s=cfg.adapter_backoff_s,
        )
    elif require_live:
        raise LiveModelUnavailable("--require-live was given but the config is in fixture mode")
    out = root / cfg.output_dir
    for sub in ("traces", "ledgers"):
        shutil.rmtree(out / sub, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    settings = cfg.episode
    if cfg.mode == "fixture":
        settings = settings.model_copy(update={"record_timing": False})
    cands, mhash = load_candidates(root / cfg.candidates)
    tuned = load_tuned(root / cfg.tuned_params)
    hashes = {"candidates": mhash, "tuned_params": _sha(root / cfg.tuned_params)}
    rows: list[dict[str, Any]] = []
    tok_in = tok_out = 0
    reported = 0.0
    p_hash = None

    def adapter(family: str, kind: str, cid: str) -> ModelAdapter:
        if live is not None:
            return live
        return FixtureAdapter.from_file(root / cfg.fixture_scripts / f"{family}_{kind}.json", cid)

    for family in cfg.families:
        specs = load_campaigns(root / "evaluator" / f"campaigns_{family}.json")
        hashes[f"{family}/campaigns"] = _sha(root / "evaluator" / f"campaigns_{family}.json")
        if cfg.mode == "fixture":
            for kind in ("agent", "single_call"):
                hashes[f"{family}/script/{kind}"] = _sha(
                    root / cfg.fixture_scripts / f"{family}_{kind}.json"
                )
        if p_hash is None:
            p_hash = prompt_hash(registry_for(make_campaign(specs[0], cands, 0), "agent"))
        for run in range(cfg.runs):
            for spec in specs:
                for arm in cfg.arms:
                    caps = cfg.run_caps
                    spent = cost_usd(tok_in, tok_out, caps)
                    capped = (
                        caps.max_total_tokens is not None
                        and tok_in + tok_out >= caps.max_total_tokens
                    ) or (caps.max_usd is not None and spent is not None and spent >= caps.max_usd)
                    name = f"{spec.campaign_id}_run{run}"
                    if arm == "random_feasible":
                        res = run_random(spec, cands, mhash, run)
                    elif arm == "cost_aware":
                        res = run_cost_aware(spec, cands, mhash, run)
                    elif arm == "tuned_acquisition":
                        res = run_tuned(spec, cands, mhash, run, tuned)
                    elif capped:
                        from .agent import new_state

                        st = new_state(spec, mhash, arm, run, 0)
                        st.status, st.stop_reason = "failed", "max_tokens"
                        res = EpisodeResult(
                            spec.campaign_id, arm, run, "failed", "run_budget_exhausted", st,
                            {"model_calls": 0},
                        )  # fmt: skip
                    elif arm == "single_call":
                        res = run_single_call(
                            spec, cands, mhash, adapter(family, "single_call", spec.campaign_id),
                            run, settings,
                        )  # fmt: skip
                    else:
                        base = root / cfg.checkpoint_dir / cfg.mode / family / arm
                        ckpt, journal = base / f"{name}.ckpt.json", base / f"{name}.journal.jsonl"
                        for p in (ckpt, journal):
                            p.unlink(missing_ok=True)
                        res = run_agent(
                            spec, cands, mhash, adapter(family, "agent", spec.campaign_id), arm,
                            run, settings,
                            trace_path=out / "traces" / family / arm / f"{name}.jsonl",
                            checkpoint_path=ckpt, journal_path=journal,
                        )  # fmt: skip
                    tok_in += res.info.get("prompt_tokens") or 0
                    tok_out += res.info.get("completion_tokens") or 0
                    reported += res.info.get("reported_cost_usd") or 0.0
                    s = score(spec, cands, res)
                    ledger_path = out / "ledgers" / family / arm / f"{name}.json"
                    ledger_path.parent.mkdir(parents=True, exist_ok=True)
                    ledger_path.write_text(
                        json.dumps(
                            {
                                "campaign_id": spec.campaign_id,
                                "arm": arm,
                                "run": run,
                                "budget_units": res.state.budget_units,
                                "spent_units": res.state.spent_units,
                                "chosen": res.state.chosen,
                                "receipts": [r.model_dump() for r in res.state.receipts],
                                "observations": [o.model_dump() for o in res.state.observations],
                            },
                            indent=1,
                        )
                        + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    info = {k: v for k, v in res.info.items() if k != "plan"}
                    rows.append(
                        {
                            "family": family,
                            "arm": arm,
                            "campaign_id": spec.campaign_id,
                            "shape": spec.shape,
                            "run": run,
                            "mode": cfg.mode,
                            "status": res.status,
                            "stop_reason": res.stop_reason,
                            "plan": res.info.get("plan"),
                            **s,
                            **info,
                        }
                    )
    meta = {
        "mode": cfg.mode,
        "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(
            _git(root, "status", "--porcelain", "--", "src", "configs", "evaluator", "tests")
        ),
        "package_version": __version__,
        "agentcore_version": AGENTCORE_VERSION,
        "model": live.name if live else "none (scripted fixture adapter)",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": p_hash,
        "config_sha256": cfg_sha,
        "config": cfg.model_dump(),
        "tuned_params": tuned.model_dump(),
        "input_hashes": hashes,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "tokens": {"prompt": tok_in, "completion": tok_out} if live else None,
        "usd": cost_usd(tok_in, tok_out, cfg.run_caps) if live else None,
        "usd_reported_by_provider": round(reported, 6) if live else None,
        "adapter_retries": getattr(live, "retries", None),
    }
    write_outputs(out, rows, meta)
    return out


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One comparison row per (family, arm), with denominators and spread."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    pending: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r["family"], r["arm"])
        if r.get("stop_reason") == PENDING_STOP:
            pending[key] = pending.get(key, 0) + 1
            groups.setdefault(key, [])
            continue
        groups.setdefault(key, []).append(r)
    out = []
    for (family, arm), rs in groups.items():
        if not rs:
            out.append({"family": family, "arm": arm, "episodes": 0,
                        "pending": pending.get((family, arm), 0)})  # fmt: skip
            continue
        reg = np.array([r["normalized_regret"] for r in rs])
        per_run: dict[int, list[float]] = {}
        for r in rs:
            per_run.setdefault(r["run"], []).append(r["normalized_regret"])
        run_means = [float(np.mean(v)) for v in per_run.values()]
        row: dict[str, Any] = {
            "family": family,
            "arm": arm,
            "mode": rs[0]["mode"],
            "episodes": len(rs),
            "runs": len(per_run),
            "pending": pending.get((family, arm), 0),
            "regret_mean": round(float(reg.mean()), 3),
            "regret_median": round(float(np.median(reg)), 3),
            "regret_run_min": round(min(run_means), 3),
            "regret_run_max": round(max(run_means), 3),
        }
        for name, key in (
            ("hit", "hit"),
            ("chosen_feasible", "chosen_feasible"),
            ("confirmed_choice", "confirmed_choice"),
        ):
            row[f"{name}_k"] = sum(r[key] for r in rs)
            row[f"{name}_n"] = len(rs)
        for key in (
            "infeasible_attempts",
            "refused_over_budget",
            "budget_violation",
            "replications",
            "confirms",
            "screens",
            "failed",
            "tool_calls",
        ):
            row[f"{key}_total"] = sum(r.get(key) or 0 for r in rs)
        row["unused_units_mean"] = round(float(np.mean([r["unused_units"] for r in rs])), 2)
        for key in ("model_calls", "malformed_actions", "reprompts"):
            row[f"{key}_total"] = sum(r.get(key) or 0 for r in rs)
        toks = [r.get("prompt_tokens") for r in rs]
        row["tokens_total"] = (
            None
            if all(t is None for t in toks)
            else sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in rs)
        )
        costs = [r.get("reported_cost_usd") for r in rs]
        row["reported_cost_usd_total"] = (
            None if all(c is None for c in costs) else round(sum(c or 0 for c in costs), 6)
        )
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        row["latency_s_mean"] = round(float(np.mean(lat)), 2) if lat else None
        out.append(row)
    return out


def paired(rows: list[dict[str, Any]], reference: str = "tuned_acquisition") -> list[dict]:
    """Win, tie and loss counts of each arm against the reference, per campaign and run."""
    ref = {
        (r["family"], r["campaign_id"], r["run"]): r["normalized_regret"]
        for r in rows
        if r["arm"] == reference
    }
    out: dict[tuple[str, str], dict[str, int]] = {}
    for r in rows:
        key = (r["family"], r["campaign_id"], r["run"])
        if r["arm"] == reference or key not in ref or r.get("stop_reason") == PENDING_STOP:
            continue
        d = r["normalized_regret"] - ref[key]
        c = out.setdefault((r["family"], r["arm"]), {"better": 0, "tie": 0, "worse": 0})
        c["better" if d < -TIE else "worse" if d > TIE else "tie"] += 1
    return [{"family": f, "arm": a, "reference": reference, **c} for (f, a), c in out.items()]


def write_outputs(out: Path, rows: list[dict], meta: dict) -> None:
    """Write task_results.jsonl, comparison.csv, paired.json and run_meta.json."""
    with (out / "task_results.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    comparison = aggregate(rows)
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = list(dict.fromkeys(k for row in comparison for k in row))
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(comparison)
    (out / "paired.json").write_text(
        json.dumps(paired(rows), indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "run_meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
