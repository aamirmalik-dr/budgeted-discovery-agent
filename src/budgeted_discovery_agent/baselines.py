"""Non-LLM baselines and the single-call LLM planner.

All arms use the same Campaign tools and ledger as the agent, so spend,
receipts and infeasible attempts are counted identically.

- random_feasible: random feasible screens, then one confirm of the best screen.
- cost_aware: space-filling (farthest-point) feasible screens, then confirm the
  top two screens. Deterministic.
- tuned_acquisition: GP-UCB screening with lengthscale, beta and the number of
  confirms tuned on the dev family only (configs/tuned_acquisition.yaml).
- single_call: one model reply chooses an allocation (strategy, screens,
  replications, confirms) that a fixed executor runs without feedback.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .agent import EpisodeResult, EpisodeSettings, llm_info, make_campaign, new_state
from .agentcore.adapters import AdapterError, ModelAdapter
from .agentcore.policy import ToolFailure, ToolRegistry
from .environment import CampaignSpec
from .prompts import brief_text, planner_system_prompt
from .schemas import CampaignState, Candidate
from .tools import Campaign, FeasibilityArgs, FinishArgs, ProposeArgs


class TunedParams(BaseModel):
    """Frozen output of scripts/tune_acquisition.py (dev family only)."""

    lengthscale: float = 0.25
    beta: float = 1.0
    n_confirm: int = 2


class Plan(BaseModel):
    """The single-call planner's allocation."""

    screen_strategy: str = Field(pattern="^(explore|exploit|balanced)$")
    screens: int = Field(ge=0, le=66)
    replicate_top: int = Field(ge=0, le=10)
    confirm_top: int = Field(ge=0, le=10)


# ---------------------------------------------------------------- helpers


def _means(state: CampaignState, fidelity: str) -> dict[str, float]:
    vals: dict[str, list[float]] = {}
    for o in state.observations:
        if o.status == "ok" and o.fidelity == fidelity and o.value is not None:
            vals.setdefault(o.candidate_id, []).append(o.value)
    return {k: float(np.mean(v)) for k, v in vals.items()}


def declare(camp: Campaign, state: CampaignState) -> None:
    """Finish with the best confirmed candidate, else the best screened one."""
    for fid in ("confirm", "screen"):
        m = _means(state, fid)
        if m:
            best = max(sorted(m), key=m.__getitem__)
            camp.finish(state, FinishArgs(best_candidate_id=best, justification=f"best {fid} mean"))
            state.status, state.stop_reason = "complete", "finished"
            return
    state.status, state.stop_reason = "stopped", "max_tool_calls"


def _try(camp: Campaign, state: CampaignState, cid: str, fid: str) -> bool:
    if state.remaining_units < camp.brief.cost[fid] - 1e-9:
        return False
    key = f"{fid}-{cid}-{len(state.receipts) + 1}"
    try:
        camp.assay(state, cid, fid, key)  # type: ignore[arg-type]
    except ToolFailure:
        return False
    return True


def _feasible(camp: Campaign, state: CampaignState, cid: str) -> bool:
    out = camp.feasibility(state, FeasibilityArgs(candidate_ids=[cid]))
    return out["results"][0]["status"] == "allowed"


def _confirm_top(camp: Campaign, state: CampaignState, k: int) -> None:
    screen = _means(state, "screen")
    for cid in sorted(screen, key=lambda c: (-screen[c], c))[:k]:
        _try(camp, state, cid, "confirm")


def _result(
    arm: str, spec: CampaignSpec, run: int, state: CampaignState, info: dict
) -> EpisodeResult:
    return EpisodeResult(spec.campaign_id, arm, run, state.status, state.stop_reason, state, info)


# ---------------------------------------------------------------- non-LLM


def run_random(spec: CampaignSpec, cands: list[Candidate], mhash: str, run: int) -> EpisodeResult:
    """Random feasible screening, then one confirm of the best screen."""
    camp = make_campaign(spec, cands, run)
    state = new_state(spec, mhash, "random_feasible", run, 999)
    rng = np.random.default_rng([spec.seed, run, 7])
    reserve = spec.cost["confirm"]
    for i in rng.permutation(len(cands)):
        if state.remaining_units < spec.cost["screen"] + reserve - 1e-9:
            break
        cid = cands[int(i)].candidate_id
        if _feasible(camp, state, cid):
            _try(camp, state, cid, "screen")
    _confirm_top(camp, state, 1)
    declare(camp, state)
    return _result("random_feasible", spec, run, state, {"model_calls": 0})


def run_cost_aware(
    spec: CampaignSpec, cands: list[Candidate], mhash: str, run: int
) -> EpisodeResult:
    """Farthest-point feasible screening, then confirm the top two screens."""
    camp = make_campaign(spec, cands, run)
    state = new_state(spec, mhash, "cost_aware", run, 999)
    X = np.array([[c.a, c.b, c.c] for c in cands])
    chosen: list[int] = []
    reserve = 2 * spec.cost["confirm"]
    centre = int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))
    order = [centre]
    while len(order) < len(cands):
        d = np.min(((X[:, None, :] - X[order][None, :, :]) ** 2).sum(-1), axis=1)
        d[order] = -1
        order.append(int(np.argmax(d)))
    for i in order:
        if state.remaining_units < spec.cost["screen"] + reserve - 1e-9:
            break
        cid = cands[i].candidate_id
        if _feasible(camp, state, cid) and _try(camp, state, cid, "screen"):
            chosen.append(i)
    _confirm_top(camp, state, 2)
    declare(camp, state)
    return _result("cost_aware", spec, run, state, {"model_calls": 0})


def run_tuned(
    spec: CampaignSpec, cands: list[Candidate], mhash: str, run: int, params: TunedParams
) -> EpisodeResult:
    """GP-UCB screening with tuned parameters, then confirm the top screens."""
    camp = make_campaign(spec, cands, run, lengthscale=params.lengthscale, beta=params.beta)
    state = new_state(spec, mhash, "tuned_acquisition", run, 999)
    reserve = params.n_confirm * spec.cost["confirm"]
    while state.remaining_units >= spec.cost["screen"] + reserve - 1e-9:
        screened = {o.candidate_id for o in state.observations}
        prohibited = {c for c, v in state.feasibility.items() if v == "prohibited"}
        ranked = camp.gp.rank(state.observations, "balanced", 1, screened | prohibited, params.beta)
        if not ranked:
            break
        cid = ranked[0][0]
        if _feasible(camp, state, cid):
            _try(camp, state, cid, "screen")
    mean, _ = camp.gp.fit(state.observations)
    pos = {cid: i for i, cid in enumerate(camp.gp.ids)}
    screen = _means(state, "screen")
    for cid in sorted(screen, key=lambda c: (-mean[pos[c]], c))[: params.n_confirm]:
        _try(camp, state, cid, "confirm")
    declare(camp, state)
    return _result("tuned_acquisition", spec, run, state, {"model_calls": 0})


# ---------------------------------------------------------------- single call


def parse_plan(text: str, max_chars: int = 4000) -> tuple[Plan | None, str | None]:
    """Parse the planner's reply into a ``Plan``."""
    if len(text) > max_chars:
        return None, f"reply was {len(text)} characters, the limit is {max_chars}"
    start = text.find("{")
    if start < 0:
        return None, "no JSON object found"
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        return Plan.model_validate({k: obj.get(k) for k in Plan.model_fields}), None
    except (json.JSONDecodeError, AttributeError) as exc:
        return None, f"invalid JSON: {exc}"
    except ValueError as exc:
        return None, f"schema error: {str(exc).splitlines()[0]}"


def execute_plan(camp: Campaign, state: CampaignState, plan: Plan) -> None:
    """Run a planner allocation open-loop, then declare."""
    for _ in range(plan.screens):
        if state.remaining_units < camp.brief.cost["screen"] - 1e-9:
            break
        screened = {o.candidate_id for o in state.observations}
        picked = None
        for _attempt in range(len(camp.cands)):
            props = camp.propose(state, ProposeArgs(strategy=plan.screen_strategy, n=5))
            fresh = [
                p["candidate_id"] for p in props["candidates"] if p["candidate_id"] not in screened
            ]
            if not fresh:
                break
            for cid in fresh:
                if _feasible(camp, state, cid):
                    picked = cid
                    break
            if picked:
                break
        if picked is None:
            break
        _try(camp, state, picked, "screen")
    screen = _means(state, "screen")
    for cid in sorted(screen, key=lambda c: (-screen[c], c))[: plan.replicate_top]:
        _try(camp, state, cid, "screen")
    _confirm_top(camp, state, plan.confirm_top)
    declare(camp, state)


def run_single_call(
    spec: CampaignSpec,
    cands: list[Candidate],
    mhash: str,
    adapter: ModelAdapter,
    run: int,
    settings: EpisodeSettings,
) -> EpisodeResult:
    """One model reply sets the allocation; a fixed executor runs it."""
    camp = make_campaign(spec, cands, run)
    registry = ToolRegistry(camp.specs())
    state = new_state(spec, mhash, "single_call", run, 999)
    messages = [
        {"role": "system", "content": planner_system_prompt(registry)},
        {"role": "user", "content": "Campaign brief:\n" + brief_text(spec.brief())},
    ]
    plan: Plan | None = None
    for attempt in range(2):
        try:
            resp = adapter.complete(messages, settings.max_completion_tokens)
        except AdapterError:
            state.status, state.stop_reason = "failed", "adapter_error"
            return _result("single_call", spec, run, state, llm_info(state, adapter.is_live, False))
        state.model_calls += 1
        state.token_usage.prompt += resp.prompt_tokens or 0
        state.token_usage.completion += resp.completion_tokens or 0
        state.reported_cost_usd += resp.cost_usd or 0.0
        plan, error = parse_plan(resp.text)
        if plan is not None:
            break
        state.reprompts += int(attempt == 0)
        messages += [
            {"role": "assistant", "content": resp.text[:4000]},
            {"role": "user", "content": f"Your reply could not be used: {error}. Try again."},
        ]
    info = llm_info(state, adapter.is_live, settings.record_timing)
    if plan is None:
        state.malformed_actions += 1
        state.status, state.stop_reason = "failed", "malformed_output"
        return _result("single_call", spec, run, state, info | {"malformed_actions": 1})
    execute_plan(camp, state, plan)
    return _result("single_call", spec, run, state, info | {"plan": plan.model_dump()})


def plan_summary(result: EpisodeResult) -> dict[str, Any] | None:
    """The allocation a single-call episode used, if any."""
    return result.info.get("plan")
