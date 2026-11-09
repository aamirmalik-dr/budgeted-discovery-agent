"""Typed records: candidates, observations, the cost ledger and campaign state."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .agentcore.schemas import BaseAgentState

Fidelity = Literal["screen", "confirm"]


class Candidate(BaseModel):
    """A fictional composition-like candidate on a three-component simplex."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    a: float
    b: float
    c: float
    label: str

    def x(self) -> tuple[float, float]:
        """Two free coordinates (c is 1 - a - b)."""
        return (self.a, self.b)


class Observation(BaseModel):
    """One committed assay result, as recorded in the ledger."""

    observation_id: str
    candidate_id: str
    fidelity: Fidelity
    value: float | None
    status: Literal["ok", "failed"]
    replicate_of: str | None = None
    receipt_id: str


class Reservation(BaseModel):
    """Units set aside for an assay that has not been committed yet."""

    reservation_id: str
    key: str
    units: float


class Receipt(BaseModel):
    """Proof of one committed charge. Receipts sum exactly to spent units."""

    receipt_id: str
    key: str
    candidate_id: str
    fidelity: Fidelity
    reserved: float
    charged: float
    refunded: float
    status: Literal["ok", "failed"]
    observation_id: str


class CampaignBrief(BaseModel):
    """Everything the agent is told about a campaign before it starts."""

    campaign_id: str
    family: Literal["dev", "heldout", "heldout_v2"]
    budget_units: float
    cost: dict[str, float] = Field(description="assay units per fidelity")
    candidates: str = "66 candidates, ids C-001 to C-066"
    failed_assay_charge_fraction: float = 0.5
    objective: str = "find the feasible candidate with the highest confirm-fidelity synthetic_score"


class CampaignState(BaseAgentState):
    """Agent-visible state. Hidden surfaces, noise streams and optima never enter it.

    The ledger (reservations, receipts, idempotency keys) lives here so that a
    checkpoint captures it and resume reuses receipts instead of re-measuring.
    """

    campaign_id: str
    candidate_manifest_hash: str
    budget_units: float
    spent_units: float = 0.0
    reserved_units: float = 0.0
    observations: list[Observation] = Field(default_factory=list)
    receipts: list[Receipt] = Field(default_factory=list)
    reservations: list[Reservation] = Field(default_factory=list)
    feasibility: dict[str, str] = Field(default_factory=dict)
    proposed_ids: list[str] = Field(default_factory=list)
    infeasible_attempts: int = 0
    refused_over_budget: int = 0
    last_error: str | None = None
    chosen: str | None = None

    @property
    def remaining_units(self) -> float:
        """Budget not yet spent or reserved."""
        return round(self.budget_units - self.spent_units - self.reserved_units, 9)

    def observable(self) -> dict[str, Any]:
        """Compact summary shown to the model each step."""
        return {
            "steps_used": self.steps,
            "tool_calls_left": self.remaining_tool_calls,
            "budget_units": self.budget_units,
            "spent_units": self.spent_units,
            "remaining_units": self.remaining_units,
            "observations": len(self.observations),
        }
