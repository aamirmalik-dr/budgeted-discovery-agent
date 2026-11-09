from __future__ import annotations

import json
from pathlib import Path

import pytest

from budgeted_discovery_agent.accounting import Ledger, LedgerError, verify
from budgeted_discovery_agent.agent import make_campaign, new_state
from budgeted_discovery_agent.agentcore.policy import ToolFailure
from budgeted_discovery_agent.candidates import load_candidates, write_manifest
from budgeted_discovery_agent.tools import FeasibilityArgs, MeasureArgs, ReplicateArgs

from .conftest import allowed_id, prohibited_id


def test_successful_assay_charges_full_cost_and_issues_receipt(boundary) -> None:
    spec, camp, st = boundary
    cid = allowed_id(camp)
    out = camp.measure(st, MeasureArgs(candidate_id=cid, fidelity="confirm", idempotency_key="a"))
    assert out["status"] == "ok" and out["cost_charged"] == spec.cost["confirm"]
    assert st.spent_units == spec.cost["confirm"] and st.reserved_units == 0
    assert st.receipts[0].refunded == 0 and verify(st) == []


def test_failed_assay_charges_half_and_refunds_the_rest(boundary) -> None:
    spec, camp, st = boundary
    out = camp.measure(
        st, MeasureArgs(candidate_id=prohibited_id(camp), fidelity="confirm", idempotency_key="b")
    )
    assert (
        out["status"] == "failed"
        and out["value"] is None
        and "assay failed" in out["failure_reason"]
    )
    rec = st.receipts[0]
    assert rec.charged == spec.cost["confirm"] / 2 and rec.refunded == spec.cost["confirm"] / 2
    assert st.infeasible_attempts == 1 and verify(st) == []


def test_duplicate_idempotency_key_charges_once(boundary) -> None:
    _, camp, st = boundary
    cid = allowed_id(camp)
    args = MeasureArgs(candidate_id=cid, fidelity="screen", idempotency_key="same")
    first, again = camp.measure(st, args), camp.measure(st, args)
    assert again["deduplicated"] and again["receipt_id"] == first["receipt_id"]
    assert again["value"] == first["value"] and len(st.receipts) == 1
    other = MeasureArgs(candidate_id=cid, fidelity="confirm", idempotency_key="same")
    with pytest.raises(ToolFailure) as info:
        camp.measure(st, other)
    assert info.value.code == "precondition_failed"


def test_insufficient_funds_is_refused_without_any_charge(boundary) -> None:
    spec, camp, st = boundary
    cid = allowed_id(camp)
    n = int(spec.budget_units // spec.cost["confirm"])
    for i in range(n):
        camp.measure(st, MeasureArgs(candidate_id=cid, fidelity="confirm", idempotency_key=f"c{i}"))
    before = (st.spent_units, len(st.receipts))
    with pytest.raises(ToolFailure) as info:
        camp.measure(st, MeasureArgs(candidate_id=cid, fidelity="confirm", idempotency_key="over"))
    assert info.value.code == "budget_exhausted"
    assert (st.spent_units, len(st.receipts)) == before and st.refused_over_budget == 1
    assert st.remaining_units >= 0 and verify(st) == []


def test_reserve_rejects_nonpositive_and_unknown_reservations(boundary) -> None:
    _, _, st = boundary
    ledger = Ledger()
    with pytest.raises(LedgerError):
        ledger.reserve(st, "k", 0)
    res = ledger.reserve(st, "k", 2)
    ledger.commit(st, res, "C-001", "screen", 0.1)
    with pytest.raises(LedgerError):
        ledger.commit(st, res, "C-001", "screen", 0.1)


def test_verify_detects_tampered_ledger(boundary) -> None:
    _, camp, st = boundary
    camp.measure(
        st, MeasureArgs(candidate_id=allowed_id(camp), fidelity="screen", idempotency_key="t")
    )
    st.spent_units += 1
    assert any("receipts sum" in p for p in verify(st))


def test_replicate_repeats_same_candidate_and_fidelity(boundary) -> None:
    _, camp, st = boundary
    cid = allowed_id(camp)
    first = camp.measure(st, MeasureArgs(candidate_id=cid, fidelity="screen", idempotency_key="r1"))
    rep = camp.replicate(
        st, ReplicateArgs(observation_id=first["observation_id"], idempotency_key="r2")
    )
    assert rep["candidate_id"] == cid and rep["fidelity"] == "screen"
    assert rep["value"] != first["value"]  # independent noise draw
    assert st.observations[-1].replicate_of == first["observation_id"]


def test_journal_lets_resume_reuse_a_committed_receipt(cands, dev_specs, tmp_path: Path) -> None:
    """Crash after the assay committed but before the agent checkpoint was written."""
    cs, mhash = cands
    spec = dev_specs["dev-smooth"]
    journal = tmp_path / "j.jsonl"
    camp = make_campaign(spec, cs, 0, journal=journal)
    checkpointed = new_state(spec, mhash, "agent", 0, 30)
    live = checkpointed.model_copy(deep=True)
    cid = allowed_id(camp)
    args = MeasureArgs(candidate_id=cid, fidelity="confirm", idempotency_key="k1")
    first = camp.measure(live, args)
    # The process dies here; resume starts again from the older checkpoint.
    resumed = make_campaign(spec, cs, 0, journal=journal)
    again = resumed.measure(checkpointed, args)
    assert again["deduplicated"] and again["receipt_id"] == first["receipt_id"]
    assert checkpointed.spent_units == spec.cost["confirm"] and len(checkpointed.receipts) == 1
    assert len(journal.read_text().splitlines()) == 1 and verify(checkpointed) == []


def test_feasibility_check_costs_nothing_and_rejects_unknown_ids(boundary) -> None:
    _, camp, st = boundary
    out = camp.feasibility(
        st, FeasibilityArgs(candidate_ids=[prohibited_id(camp), allowed_id(camp)])
    )
    assert [r["status"] for r in out["results"]] == ["prohibited", "allowed"]
    assert st.spent_units == 0 and "rule" in out["results"][0]["reason"]
    with pytest.raises(ToolFailure):
        camp.feasibility(st, FeasibilityArgs(candidate_ids=["C-999"]))


def test_changed_manifest_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    write_manifest(path)
    load_candidates(path)
    data = json.loads(path.read_text())
    data["candidates"][0]["a"] = 0.05
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest"):
        load_candidates(path)
