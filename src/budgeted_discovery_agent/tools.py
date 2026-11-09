"""The seven campaign tools. Every assay goes through the ledger.

Tools return observable quantities only: noisy values, surrogate predictions
fitted on observed data, the feasibility verdict and its reason, and ledger
balances. None returns a true value, a noise level or the optimum.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .accounting import Ledger, LedgerError
from .agentcore.policy import ToolFailure, ToolSpec
from .environment import Simulator
from .schemas import CampaignBrief, CampaignState, Candidate, Fidelity
from .surrogate import DEFAULT_LENGTHSCALE, GPSurrogate

ALL_TOOLS = (
    "propose_candidates",
    "surrogate_summary",
    "check_feasibility",
    "measure",
    "replicate",
    "remaining_budget",
    "finish",
)
LABEL_CHARS = 200


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProposeArgs(_Args):
    strategy: Literal["explore", "exploit", "balanced"]
    n: int = Field(default=3, ge=1, le=5)


class NoArgs(_Args):
    pass


class FeasibilityArgs(_Args):
    candidate_ids: list[str] = Field(min_length=1, max_length=10)


class MeasureArgs(_Args):
    candidate_id: str = Field(min_length=1, max_length=20)
    fidelity: Fidelity
    idempotency_key: str = Field(
        min_length=1, max_length=64, description="reuse the same key only to retry the same assay"
    )


class ReplicateArgs(_Args):
    observation_id: str = Field(min_length=1, max_length=20)
    idempotency_key: str = Field(min_length=1, max_length=64)


class FinishArgs(_Args):
    best_candidate_id: str = Field(min_length=1, max_length=20)
    justification: str = Field(min_length=1, max_length=400)


class Campaign:
    """Binds a simulator, the manifest, the brief and the ledger for one episode.

    Args:
        sim: The simulated lab for this campaign and run.
        candidates: The candidate manifest.
        brief: Visible campaign brief (budget and costs).
        ledger: Ledger, optionally journalled.
        lengthscale: Surrogate lengthscale used by propose and summary.
        beta: UCB weight of the balanced strategy.
        selector_override: If set, propose ignores the requested strategy and
            uses this one (the default-selector ablation).
    """

    def __init__(
        self,
        sim: Simulator,
        candidates: list[Candidate],
        brief: CampaignBrief,
        ledger: Ledger | None = None,
        lengthscale: float = DEFAULT_LENGTHSCALE,
        beta: float = 1.0,
        selector_override: str | None = None,
    ) -> None:
        self.sim, self.brief = sim, brief
        self.cands = {c.candidate_id: c for c in candidates}
        self.ledger = ledger or Ledger()
        self.gp = GPSurrogate(candidates, lengthscale)
        self.beta = beta
        self.selector_override = selector_override

    # ------------------------------------------------------------ helpers

    def _known(self, candidate_id: str) -> None:
        if candidate_id not in self.cands:
            raise ToolFailure(
                "not_found", f"no candidate {candidate_id!r}; ids run from C-001 to C-066"
            )

    def assay(
        self,
        state: CampaignState,
        candidate_id: str,
        fidelity: Fidelity,
        key: str,
        replicate_of: str | None = None,
    ) -> dict[str, Any]:
        """Reserve, run and commit one assay exactly once per idempotency key."""
        done = self.ledger.find(state, key)
        if done is not None:
            rec, obs = done
            if (rec.candidate_id, rec.fidelity) != (candidate_id, fidelity):
                raise ToolFailure(
                    "precondition_failed", f"idempotency key {key!r} was used for another assay"
                )
            return self._result(state, rec, obs, deduplicated=True)
        cost = self.brief.cost[fidelity]
        try:
            res = self.ledger.reserve(state, key, cost)
        except LedgerError as exc:
            state.refused_over_budget += 1
            state.last_error = str(exc)
            raise ToolFailure("budget_exhausted", str(exc)) from None
        allowed, reason = self.sim.feasibility(candidate_id)
        state.feasibility[candidate_id] = "allowed" if allowed else "prohibited"
        if allowed:
            k = sum(
                1
                for o in state.observations
                if o.candidate_id == candidate_id and o.fidelity == fidelity
            )
            value: float | None = self.sim.measure(candidate_id, fidelity, k)
        else:
            value = None
            state.infeasible_attempts += 1
        rec, obs = self.ledger.commit(state, res, candidate_id, fidelity, value, replicate_of)
        out = self._result(state, rec, obs, deduplicated=False)
        if not allowed:
            out["failure_reason"] = f"assay failed: {reason}"
        return out

    def _result(self, state: CampaignState, rec: Any, obs: Any, deduplicated: bool) -> dict:
        return {
            "observation_id": obs.observation_id,
            "candidate_id": obs.candidate_id,
            "fidelity": obs.fidelity,
            "status": obs.status,
            "value": obs.value,
            "units": "synthetic_score",
            "cost_charged": rec.charged,
            "receipt_id": rec.receipt_id,
            "remaining_units": state.remaining_units,
            "deduplicated": deduplicated,
        }

    # ------------------------------------------------------------ tools

    def propose(self, state: CampaignState, a: ProposeArgs) -> dict[str, Any]:
        strategy = self.selector_override or a.strategy
        exclude = {cid for cid, v in state.feasibility.items() if v == "prohibited"}
        ranked = self.gp.rank(state.observations, strategy, a.n, exclude, self.beta)
        for cid, _, _ in ranked:
            if cid not in state.proposed_ids:
                state.proposed_ids.append(cid)
        return {
            "strategy_used": strategy,
            "candidates": [
                {
                    "candidate_id": cid,
                    "label": self.cands[cid].label[:LABEL_CHARS],
                    "predicted_mean": round(m, 4),
                    "uncertainty": round(s, 4),
                    "feasibility": state.feasibility.get(cid, "unchecked"),
                    "estimated_cost": dict(self.brief.cost),
                }
                for cid, m, s in ranked
            ],
        }

    def summary(self, state: CampaignState, a: NoArgs) -> dict[str, Any]:
        mean, sd = self.gp.fit(state.observations)
        ids = self.gp.ids
        exclude = {cid for cid, v in state.feasibility.items() if v == "prohibited"}
        top = sorted(
            (i for i in range(len(ids)) if ids[i] not in exclude), key=lambda i: (-mean[i], ids[i])
        )[:3]
        ok = [o for o in state.observations if o.status == "ok"]
        per: dict[tuple[str, str], list[float]] = {}
        for o in ok:
            per.setdefault((o.candidate_id, o.fidelity), []).append(o.value or 0.0)
        return {
            "observations_ok": len(ok),
            "observations_failed": len(state.observations) - len(ok),
            "lengthscale": self.gp.lengthscale,
            "top_by_posterior_mean": [
                {
                    "candidate_id": ids[i],
                    "mean": round(float(mean[i]), 4),
                    "sd": round(float(sd[i]), 4),
                }
                for i in top
            ],
            "observed_means": [
                {"candidate_id": c, "fidelity": f, "n": len(v), "mean": round(sum(v) / len(v), 4)}
                for (c, f), v in sorted(per.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))[:6]
            ],
            "calibration": self.gp.calibration(state.observations),
        }

    def feasibility(self, state: CampaignState, a: FeasibilityArgs) -> dict[str, Any]:
        results = []
        for cid in dict.fromkeys(a.candidate_ids):
            self._known(cid)
            allowed, reason = self.sim.feasibility(cid)
            state.feasibility[cid] = "allowed" if allowed else "prohibited"
            results.append(
                {"candidate_id": cid, "status": state.feasibility[cid], "reason": reason}
            )
        return {"results": results}

    def measure(self, state: CampaignState, a: MeasureArgs) -> dict[str, Any]:
        self._known(a.candidate_id)
        return self.assay(state, a.candidate_id, a.fidelity, a.idempotency_key)

    def replicate(self, state: CampaignState, a: ReplicateArgs) -> dict[str, Any]:
        src = next((o for o in state.observations if o.observation_id == a.observation_id), None)
        if src is None:
            raise ToolFailure("not_found", f"no observation {a.observation_id!r}")
        if src.status != "ok":
            raise ToolFailure("precondition_failed", "only a successful assay can be replicated")
        return self.assay(
            state,
            src.candidate_id,
            src.fidelity,
            a.idempotency_key,
            replicate_of=src.observation_id,
        )

    def budget(self, state: CampaignState, a: NoArgs) -> dict[str, Any]:
        return {
            "budget_units": state.budget_units,
            "spent_units": state.spent_units,
            "reserved_units": state.reserved_units,
            "remaining_units": state.remaining_units,
            "receipts": len(state.receipts),
            "receipts_sum": round(sum(r.charged for r in state.receipts), 9),
            "cost": dict(self.brief.cost),
        }

    def finish(self, state: CampaignState, a: FinishArgs) -> dict[str, Any]:
        self._known(a.best_candidate_id)
        state.chosen = a.best_candidate_id
        return {"best_candidate_id": a.best_candidate_id, "justification": a.justification}

    def specs(self) -> list[ToolSpec]:
        """Tool specs for the registry."""
        return [
            ToolSpec(
                "propose_candidates",
                "Rank candidates with the surrogate fitted on your observations. strategy: "
                "explore (highest uncertainty), exploit (highest predicted mean), balanced "
                "(mean plus uncertainty). Skips candidates known to be prohibited.",
                ProposeArgs,
                self.propose,
            ),
            ToolSpec(
                "surrogate_summary",
                "Surrogate fit diagnostics, top candidates by posterior mean, observed means.",
                NoArgs,
                self.summary,
            ),
            ToolSpec(
                "check_feasibility",
                "Allowed or prohibited, with the reason, for up to 10 candidates. Costs no "
                "assay units.",
                FeasibilityArgs,
                self.feasibility,
            ),
            ToolSpec(
                "measure",
                "Run one assay at screen (cheap, noisier, may be biased) or confirm (expensive, "
                "accurate) fidelity. Charges the budget once per idempotency key.",
                MeasureArgs,
                self.measure,
                side_effect=True,
            ),
            ToolSpec(
                "replicate",
                "Independent repeat of an earlier successful observation at the same fidelity "
                "and cost.",
                ReplicateArgs,
                self.replicate,
                side_effect=True,
            ),
            ToolSpec("remaining_budget", "Ledger balances and receipt count.", NoArgs, self.budget),
            ToolSpec(
                "finish",
                "End the campaign and declare the best feasible candidate.",
                FinishArgs,
                self.finish,
                terminal="complete",
            ),
        ]
