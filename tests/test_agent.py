from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from budgeted_discovery_agent.agent import EpisodeSettings, make_campaign, new_state, run_agent
from budgeted_discovery_agent.agentcore.adapters import FixtureAdapter
from budgeted_discovery_agent.agentcore.loop import CheckpointStore
from budgeted_discovery_agent.agentcore.trace import read_trace
from budgeted_discovery_agent.candidates import INJECTED_ID
from budgeted_discovery_agent.environment import Simulator
from budgeted_discovery_agent.prompts import make_renderer
from budgeted_discovery_agent.schemas import CampaignState
from budgeted_discovery_agent.surrogate import GPSurrogate
from budgeted_discovery_agent.tools import ALL_TOOLS, ProposeArgs

from .conftest import ROOT


def act(tool: str, summary: str = "s", **arguments) -> dict:
    return {"summary": summary, "action": {"tool": tool, "arguments": arguments}}


def run(spec, cands, script, tmp_path: Path, arm="agent", **kw):
    cs, mhash = cands
    return run_agent(
        spec, cs, mhash, FixtureAdapter(script), arm, 0, EpisodeSettings(record_timing=False),
        trace_path=tmp_path / "t.jsonl", checkpoint_path=tmp_path / "c.ckpt.json",
        journal_path=tmp_path / "j.jsonl", **kw,
    )  # fmt: skip


def test_tools_expose_no_oracle(cands, dev_specs) -> None:
    cs, mhash = cands
    camp = make_campaign(dev_specs["dev-smooth"], cs, 0)
    assert {s.name for s in camp.specs()} == set(ALL_TOOLS)
    st = new_state(dev_specs["dev-smooth"], mhash, "t", 0, 10)
    text = json.dumps(camp.propose(st, ProposeArgs(strategy="balanced", n=5)))
    for hidden in ("true", "optimum", "f_star", "bias", "noise"):
        assert hidden not in text


def test_common_random_numbers_pair_arms(cands, dev_specs) -> None:
    cs, _ = cands
    spec = dev_specs["dev-neartie"]
    a, b = Simulator(spec, cs, 0), Simulator(spec, cs, 0)
    assert a.measure("C-029", "screen", 0) == b.measure("C-029", "screen", 0)
    assert a.measure("C-029", "screen", 0) != a.measure("C-029", "screen", 1)
    assert a.measure("C-029", "screen", 0) != Simulator(spec, cs, 1).measure("C-029", "screen", 0)


def test_gp_fits_observed_data_and_skips_prohibited(cands, dev_specs, boundary) -> None:
    cs, _ = cands
    spec, camp, st = boundary
    from budgeted_discovery_agent.tools import MeasureArgs

    for i, cid in enumerate(("C-001", "C-030", "C-066", "C-045")):
        camp.measure(st, MeasureArgs(candidate_id=cid, fidelity="confirm", idempotency_key=str(i)))
    mean, sd = GPSurrogate(cs).fit(st.observations)
    obs = {o.candidate_id: o.value for o in st.observations if o.value is not None}
    pos = {c.candidate_id: i for i, c in enumerate(cs)}
    for cid, v in obs.items():
        assert abs(mean[pos[cid]] - v) < 0.1 and sd[pos[cid]] < 0.1
    assert np.all(sd > 0)
    st.feasibility["C-002"] = "prohibited"
    ranked = GPSurrogate(cs).rank(st.observations, "explore", 66, {"C-002"})
    assert "C-002" not in [r[0] for r in ranked]


def test_planted_over_budget_attempt_is_refused_and_episode_continues(
    cands, dev_specs, tmp_path
) -> None:
    spec = dev_specs["dev-smooth"]
    script = [
        act("measure", candidate_id="C-045", fidelity="confirm", idempotency_key=f"k{i}")
        for i in range(7)
    ]
    script.append(act("finish", best_candidate_id="C-045", justification="best confirm"))
    res = run(spec, cands, script, tmp_path)
    st = res.state
    assert res.status == "complete" and st.refused_over_budget == 1
    assert st.spent_units == 36 and st.remaining_units == 0
    refused = [h for h in st.history if not h.result.ok]
    assert refused[0].result.error.code == "budget_exhausted"


def test_command_like_label_stays_data(cands, dev_specs, tmp_path) -> None:
    spec = dev_specs["dev-smooth"]
    cs, mhash = cands
    camp = make_campaign(spec, cs, 0)
    st = new_state(spec, mhash, "agent", 0, 10)
    st.observations = []
    label = camp.cands[INJECTED_ID].label
    assert "NOTE TO ANY AGENT" in label
    # Proposals that include the injected candidate wrap its label as untrusted data.
    script = [act("propose_candidates", strategy="explore", n=5)] * 3
    script.append(act("finish", best_candidate_id="C-001", justification="x"))
    res = run(spec, cands, script, tmp_path)
    allowed = {
        e["action"]["tool"] for e in read_trace(tmp_path / "t.jsonl") if e["event"] == "step"
    }
    assert allowed <= set(ALL_TOOLS) and res.state.spent_units == 0
    render = make_renderer(spec.brief())
    text = render(res.state, res.state.history[-2:])
    assert 'trust="untrusted-data"' in text


def test_replicate_disabled_in_ablation(cands, dev_specs, tmp_path) -> None:
    spec = dev_specs["dev-smooth"]
    script = [
        act("measure", candidate_id="C-045", fidelity="screen", idempotency_key="a"),
        act("replicate", observation_id="O-001", idempotency_key="b"),
        act("finish", best_candidate_id="C-045", justification="x"),
    ]
    res = run(spec, cands, script, tmp_path, arm="agent_no_replicate")
    assert res.state.history[1].result.error.code == "not_allowed"
    assert len(res.state.receipts) == 1


def test_default_selector_ablation_overrides_strategy(cands, dev_specs, tmp_path) -> None:
    spec = dev_specs["dev-smooth"]
    script = [act("propose_candidates", strategy="explore", n=3),
              act("finish", best_candidate_id="C-001", justification="x")]  # fmt: skip
    res = run(spec, cands, script, tmp_path, arm="agent_default_selector")
    assert res.state.history[0].result.output["strategy_used"] == "balanced"


def test_malformed_reply_and_adapter_failure_are_recorded(cands, dev_specs, tmp_path) -> None:
    spec = dev_specs["dev-smooth"]
    res = run(spec, cands, ["not json", "still not json"], tmp_path)
    assert res.state.malformed_actions == 1
    assert res.status == "failed" and res.stop_reason == "adapter_error"  # script exhausted


def test_resume_does_not_repeat_a_committed_measurement(cands, dev_specs, tmp_path) -> None:
    spec = dev_specs["dev-smooth"]
    script = [
        act("measure", candidate_id="C-045", fidelity="screen", idempotency_key="a"),
        act("measure", candidate_id="C-046", fidelity="screen", idempotency_key="b"),
        act("finish", best_candidate_id="C-045", justification="x"),
    ]
    run(spec, cands, script[:1], tmp_path)  # adapter runs dry after one action
    store = CheckpointStore(tmp_path / "c.ckpt.json")
    st = store.load(CampaignState)
    st.status, st.stop_reason = "running", None
    store.save(st)
    res = run(spec, cands, script, tmp_path, resume=True)
    assert res.status == "complete"
    assert [r.candidate_id for r in res.state.receipts] == ["C-045", "C-046"]
    assert len((tmp_path / "j.jsonl").read_text().splitlines()) == 2


def test_truncated_checkpoint_falls_back_to_previous(cands, dev_specs, tmp_path) -> None:
    spec = dev_specs["dev-smooth"]
    script = [act("remaining_budget"), act("remaining_budget")]
    run(spec, cands, script, tmp_path)
    path = tmp_path / "c.ckpt.json"
    path.write_text(path.read_text()[:40])
    st = CheckpointStore(path).load(CampaignState)
    assert st is not None and st.campaign_id == "dev-smooth"


def test_fixture_scripts_exist_for_every_campaign() -> None:
    for family in ("dev", "heldout"):
        camps = json.loads((ROOT / "evaluator" / f"campaigns_{family}.json").read_text())
        ids = {c["campaign_id"] for c in camps["campaigns"]}
        for kind in ("agent", "single_call"):
            scripts = json.loads(
                (ROOT / "tests/fixtures/scripts" / f"{family}_{kind}.json").read_text()
            )
            assert set(scripts) == ids


def test_observation_lists_checked_allowed_candidates(cands, dev_specs) -> None:
    from budgeted_discovery_agent.tools import FeasibilityArgs

    spec = dev_specs["dev-boundary"]
    cs, mhash = cands
    camp = make_campaign(spec, cs, 0)
    st = new_state(spec, mhash, "agent", 0, 10)
    camp.feasibility(st, FeasibilityArgs(candidate_ids=["C-001", "C-044"]))
    text = make_renderer(spec.brief())(st, [])
    assert "Already checked and allowed: C-001, C-044" in text
