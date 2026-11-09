"""The cost ledger: reserve, commit, refund, receipts and idempotency keys.

Two copies are kept. The campaign state holds receipts and observations, so
the agent checkpoint carries them. An optional append-only journal file is
the authoritative record: every commit is fsynced to it before the tool
returns. If the process dies after an assay committed but before the agent
checkpoint was written, resume finds the receipt in the journal and reuses
it instead of running and charging the assay again.

Invariant, checked by ``verify``: spent + reserved + remaining == budget,
receipts sum exactly to spent, and remaining is never negative.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schemas import CampaignState, Fidelity, Observation, Receipt, Reservation

FAILED_ASSAY_CHARGE = 0.5
"""Fraction of the assay cost charged when an assay fails (infeasible candidate)."""


class LedgerError(Exception):
    """A charge the ledger refuses (insufficient funds, unknown reservation)."""


class Ledger:
    """Ledger operations on a campaign state, with an optional journal.

    Args:
        journal: Path of the append-only JSONL journal, or None for no journal.
    """

    def __init__(self, journal: Path | None = None) -> None:
        self.journal = Path(journal) if journal else None
        if self.journal:
            self.journal.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ lookup

    def find(self, state: CampaignState, key: str) -> tuple[Receipt, Observation] | None:
        """Receipt and observation already committed under ``key``, if any.

        A receipt found only in the journal is merged back into the state.
        """
        for r in state.receipts:
            if r.key == key:
                obs = next(o for o in state.observations if o.observation_id == r.observation_id)
                return r, obs
        for rec, obs in self._journal_entries():
            if rec.key == key:
                state.receipts.append(rec)
                state.observations.append(obs)
                state.spent_units = round(state.spent_units + rec.charged, 9)
                return rec, obs
        return None

    def _journal_entries(self) -> list[tuple[Receipt, Observation]]:
        if not self.journal or not self.journal.exists():
            return []
        out = []
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out.append((Receipt.model_validate(d["receipt"]), Observation(**d["observation"])))
        return out

    # ------------------------------------------------------------ charges

    def reserve(self, state: CampaignState, key: str, units: float) -> Reservation:
        """Set aside ``units``; refuse if the remaining budget cannot cover them."""
        if units <= 0:
            raise LedgerError("a reservation must be positive")
        if units > state.remaining_units + 1e-9:
            raise LedgerError(
                f"insufficient funds: {units} units needed, {state.remaining_units} remaining"
            )
        res = Reservation(
            reservation_id=f"R-{len(state.reservations) + 1:04d}", key=key, units=units
        )
        state.reservations.append(res)
        state.reserved_units = round(state.reserved_units + units, 9)
        return res

    def commit(
        self,
        state: CampaignState,
        res: Reservation,
        candidate_id: str,
        fidelity: Fidelity,
        value: float | None,
        replicate_of: str | None = None,
    ) -> tuple[Receipt, Observation]:
        """Charge a reservation: in full on success, a fixed fraction on failure.

        The unused part of the reservation is refunded. The journal entry is
        written and fsynced before the state changes.
        """
        if res not in state.reservations:
            raise LedgerError(f"unknown reservation {res.reservation_id}")
        ok = value is not None
        charged = res.units if ok else round(res.units * FAILED_ASSAY_CHARGE, 9)
        n = len(state.receipts) + 1
        obs = Observation(
            observation_id=f"O-{n:03d}",
            candidate_id=candidate_id,
            fidelity=fidelity,
            value=value,
            status="ok" if ok else "failed",
            replicate_of=replicate_of,
            receipt_id=f"RC-{n:03d}",
        )
        rec = Receipt(
            receipt_id=obs.receipt_id,
            key=res.key,
            candidate_id=candidate_id,
            fidelity=fidelity,
            reserved=res.units,
            charged=charged,
            refunded=round(res.units - charged, 9),
            status=obs.status,
            observation_id=obs.observation_id,
        )
        if self.journal:
            line = json.dumps({"receipt": rec.model_dump(), "observation": obs.model_dump()})
            with self.journal.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        state.reservations.remove(res)
        state.reserved_units = round(state.reserved_units - res.units, 9)
        state.spent_units = round(state.spent_units + charged, 9)
        state.receipts.append(rec)
        state.observations.append(obs)
        return rec, obs


def verify(state: CampaignState) -> list[str]:
    """Return every ledger invariant that does not hold (empty means consistent)."""
    problems = []
    total = round(sum(r.charged for r in state.receipts), 9)
    if abs(total - state.spent_units) > 1e-9:
        problems.append(f"receipts sum to {total}, spent is {state.spent_units}")
    if state.remaining_units < -1e-9:
        problems.append(f"remaining is negative: {state.remaining_units}")
    if (
        abs(state.spent_units + state.reserved_units + state.remaining_units - state.budget_units)
        > 1e-9
    ):
        problems.append("spent + reserved + remaining does not equal the budget")
    keys = [r.key for r in state.receipts]
    if len(keys) != len(set(keys)):
        problems.append("an idempotency key was charged twice")
    for r in state.receipts:
        if abs(r.charged + r.refunded - r.reserved) > 1e-9:
            problems.append(f"{r.receipt_id}: charged + refunded != reserved")
    return problems
