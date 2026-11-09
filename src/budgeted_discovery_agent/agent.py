"""Episode setup shared by every arm, and the iterative LLM agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .accounting import Ledger
from .agentcore.adapters import ModelAdapter
from .agentcore.loop import AgentLoop, CheckpointStore
from .agentcore.policy import Budget, Policy, ToolRegistry
from .agentcore.trace import TraceWriter
from .environment import CampaignSpec, Simulator
from .prompts import agent_system_prompt, make_renderer
from .schemas import CampaignState, Candidate
from .surrogate import DEFAULT_LENGTHSCALE
from .tools import ALL_TOOLS, Campaign

AGENT_ARMS: dict[str, dict[str, Any]] = {
    "agent": {},
    "agent_default_selector": {"selector_override": "balanced"},
    "agent_no_replicate": {"allow": tuple(t for t in ALL_TOOLS if t != "replicate")},
}


class EpisodeSettings(BaseModel):
    """Per-episode caps for the LLM agent (assay units are capped by the ledger)."""

    max_steps: int = 44
    max_tool_calls: int = 36
    max_tokens: int | None = 80000
    max_wall_seconds: float | None = 600
    recent_k: int = 4
    max_completion_tokens: int = 400
    record_timing: bool = True


@dataclass
class EpisodeResult:
    """What the evaluator needs from one campaign run of one arm."""

    campaign_id: str
    arm: str
    run: int
    status: str
    stop_reason: str | None
    state: CampaignState
    info: dict[str, Any] = field(default_factory=dict)


def new_state(
    spec: CampaignSpec, manifest_hash: str, arm: str, run: int, calls: int
) -> CampaignState:
    """Fresh campaign state. Hidden surfaces and optima are never part of it."""
    return CampaignState(
        episode_id=f"{arm}:{spec.campaign_id}:run{run}",
        campaign_id=spec.campaign_id,
        candidate_manifest_hash=manifest_hash,
        budget_units=spec.budget_units,
        remaining_tool_calls=calls,
    )


def make_campaign(
    spec: CampaignSpec,
    candidates: list[Candidate],
    run: int,
    journal: Path | None = None,
    lengthscale: float = DEFAULT_LENGTHSCALE,
    beta: float = 1.0,
    selector_override: str | None = None,
) -> Campaign:
    """Simulator, ledger and tool bundle for one campaign run."""
    return Campaign(
        Simulator(spec, candidates, run),
        candidates,
        spec.brief(),
        Ledger(journal),
        lengthscale=lengthscale,
        beta=beta,
        selector_override=selector_override,
    )


def registry_for(campaign: Campaign, arm: str) -> ToolRegistry:
    """Registry with the arm's allowlist."""
    return ToolRegistry(campaign.specs(), allowlist=AGENT_ARMS.get(arm, {}).get("allow"))


def llm_info(state: CampaignState, live: bool, record_timing: bool) -> dict[str, Any]:
    """Model accounting shared by the LLM arms."""
    return {
        "model_calls": state.model_calls,
        "steps": state.steps,
        "malformed_actions": state.malformed_actions,
        "reprompts": state.reprompts,
        "prompt_tokens": state.token_usage.prompt if live else None,
        "completion_tokens": state.token_usage.completion if live else None,
        "reported_cost_usd": round(state.reported_cost_usd, 6) if live else None,
        "latency_s": round(state.elapsed_seconds, 3) if record_timing else None,
    }


def run_agent(
    spec: CampaignSpec,
    candidates: list[Candidate],
    manifest_hash: str,
    adapter: ModelAdapter,
    arm: str,
    run: int,
    settings: EpisodeSettings,
    trace_path: Path | None = None,
    checkpoint_path: Path | None = None,
    journal_path: Path | None = None,
    resume: bool = False,
) -> EpisodeResult:
    """Run (or resume) the iterative agent on one campaign."""
    campaign = make_campaign(
        spec,
        candidates,
        run,
        journal=journal_path,
        selector_override=AGENT_ARMS[arm].get("selector_override"),
    )
    registry = registry_for(campaign, arm)
    store = CheckpointStore(checkpoint_path) if checkpoint_path else None
    state = store.load(CampaignState) if (resume and store) else None
    if state is None:
        state = new_state(spec, manifest_hash, arm, run, settings.max_tool_calls)
    loop = AgentLoop(
        adapter,
        Policy(
            registry,
            Budget(
                max_steps=settings.max_steps,
                max_tool_calls=settings.max_tool_calls,
                max_tokens=settings.max_tokens,
                max_wall_seconds=settings.max_wall_seconds,
            ),
        ),
        agent_system_prompt(registry),
        make_renderer(spec.brief()),
        checkpoint=store,
        trace=TraceWriter(trace_path, settings.record_timing) if trace_path else None,
        recent_k=settings.recent_k,
        max_completion_tokens=settings.max_completion_tokens,
    )
    state = loop.run(state)
    return EpisodeResult(
        spec.campaign_id,
        arm,
        run,
        state.status,
        state.stop_reason,
        state,
        llm_info(state, adapter.is_live, settings.record_timing),
    )
