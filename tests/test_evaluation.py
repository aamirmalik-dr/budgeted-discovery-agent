from __future__ import annotations

import pytest

from budgeted_discovery_agent.__main__ import main, run_demo
from budgeted_discovery_agent.accounting import verify
from budgeted_discovery_agent.agent import EpisodeResult, EpisodeSettings, new_state
from budgeted_discovery_agent.agentcore.adapters import FixtureAdapter
from budgeted_discovery_agent.baselines import (
    Plan,
    parse_plan,
    run_cost_aware,
    run_random,
    run_single_call,
    run_tuned,
)
from budgeted_discovery_agent.environment import Simulator, load_campaigns
from budgeted_discovery_agent.evaluation import load_tuned, paired, score

from .conftest import ROOT


def _result(spec, mhash, chosen, status="complete"):
    st = new_state(spec, mhash, "x", 0, 1)
    st.chosen = chosen
    return EpisodeResult(spec.campaign_id, "x", 0, status, None, st)


def test_score_regret_hit_and_failure_semantics(cands, dev_specs) -> None:
    cs, mhash = cands
    spec = dev_specs["dev-deceptive"]
    oracle = Simulator(spec, cs, 0).oracle_summary(cs)
    best = score(spec, cs, _result(spec, mhash, oracle["best_id"]))
    assert best["normalized_regret"] == 0 and best["hit"] == 1
    none = score(spec, cs, _result(spec, mhash, None, status="stopped"))
    assert none["normalized_regret"] == 1.0 and none["failed"] == 1
    bad = next(
        c.candidate_id for c in cs if not Simulator(spec, cs, 0).feasibility(c.candidate_id)[0]
    )
    assert score(spec, cs, _result(spec, mhash, bad))["chosen_feasible"] == 0


@pytest.mark.parametrize("family", ["dev", "heldout"])
def test_non_llm_baselines_respect_budget_on_every_campaign(cands, family) -> None:
    cs, mhash = cands
    tuned = load_tuned(ROOT / "configs" / "tuned_acquisition.yaml")
    for spec in load_campaigns(ROOT / "evaluator" / f"campaigns_{family}.json"):
        for res in (
            run_random(spec, cs, mhash, 0),
            run_cost_aware(spec, cs, mhash, 0),
            run_tuned(spec, cs, mhash, 0, tuned),
        ):
            assert verify(res.state) == [] and res.state.spent_units <= spec.budget_units
            assert res.status == "complete" and res.state.chosen is not None


def test_baselines_are_deterministic(cands, dev_specs) -> None:
    cs, mhash = cands
    spec = dev_specs["dev-neartie"]
    a, b = run_random(spec, cs, mhash, 3), run_random(spec, cs, mhash, 3)
    assert [r.model_dump() for r in a.state.receipts] == [r.model_dump() for r in b.state.receipts]


def test_single_call_plan_parsing_and_execution(cands, dev_specs) -> None:
    cs, mhash = cands
    plan, err = parse_plan('{"summary": "s", "screen_strategy": "explore", "screens": 5, '
                           '"replicate_top": 1, "confirm_top": 2}}')  # fmt: skip
    assert err is None and plan == Plan(
        screen_strategy="explore", screens=5, replicate_top=1, confirm_top=2
    )
    assert parse_plan('{"screen_strategy": "greedy", "screens": 5}')[0] is None
    spec = dev_specs["dev-smooth"]
    good = (
        '{"summary": "s", "screen_strategy": "balanced", "screens": 40, '
        '"replicate_top": 2, "confirm_top": 3}'
    )
    res = run_single_call(spec, cs, mhash, FixtureAdapter([good]), 0, EpisodeSettings())
    assert res.status == "complete" and verify(res.state) == []
    assert res.state.spent_units <= spec.budget_units
    bad = run_single_call(spec, cs, mhash, FixtureAdapter(["prose", "prose"]), 0, EpisodeSettings())
    assert bad.status == "failed" and bad.stop_reason == "malformed_output"


def test_paired_counts() -> None:
    rows = [
        {
            "family": "f",
            "campaign_id": "c",
            "run": 0,
            "arm": "tuned_acquisition",
            "normalized_regret": 0.2,
        },
        {"family": "f", "campaign_id": "c", "run": 0, "arm": "agent", "normalized_regret": 0.0},
        {
            "family": "f",
            "campaign_id": "d",
            "run": 0,
            "arm": "tuned_acquisition",
            "normalized_regret": 0.0,
        },
        {"family": "f", "campaign_id": "d", "run": 0, "arm": "agent", "normalized_regret": 0.005},
    ]
    assert paired(rows) == [
        {
            "family": "f",
            "arm": "agent",
            "reference": "tuned_acquisition",
            "better": 1,
            "tie": 1,
            "worse": 0,
        }
    ]


def test_demo_runs_offline_and_ledger_is_consistent() -> None:
    res = run_demo("fixture", ROOT)
    assert res.status == "complete" and verify(res.state) == []
    assert res.state.reprompts == 1  # the planted prose reply
    assert any(o.replicate_of for o in res.state.observations)


def test_require_live_fails_clearly(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    for var in ("AGENT_MODEL_BASE_URL", "AGENT_MODEL_NAME", "AGENT_MODEL_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(ROOT)
    assert main(["evaluate", "--config", "configs/campaign_fixture.yaml", "--require-live"]) == 2
    assert main(["evaluate", "--config", "configs/campaign_live.yaml", "--require-live"]) == 2
    assert "fixture fallback is disabled" in " ".join(capsys.readouterr().out.split())


def test_audit_reads_only_ledgers_and_catches_tampering(tmp_path) -> None:
    from budgeted_discovery_agent.__main__ import audit_ledgers

    led = tmp_path / "fixture" / "ledgers" / "dev" / "agent"
    led.mkdir(parents=True)
    (tmp_path / "fixture" / "paired.json").write_text("[]")
    rec = {"charged": 2.0, "key": "k1"}
    doc = {"budget_units": 36, "spent_units": 2.0, "receipts": [rec]}
    import json

    (led / "a.json").write_text(json.dumps(doc))
    assert audit_ledgers(tmp_path) == (1, [])
    (led / "b.json").write_text(json.dumps({**doc, "spent_units": 4.0, "receipts": [rec, rec]}))
    n, problems = audit_ledgers(tmp_path)
    assert n == 2 and any("duplicate" in p for p in problems)


def test_v2_executor_flags_replicates(cands, dev_specs) -> None:
    cs, mhash = cands
    spec = dev_specs["dev-smooth"]
    good = (
        '{"summary": "s", "screen_strategy": "balanced", "screens": 6, '
        '"replicate_top": 2, "confirm_top": 1}'
    )
    res = run_single_call(spec, cs, mhash, FixtureAdapter([good]), 0, EpisodeSettings())
    reps = [o for o in res.state.observations if o.replicate_of]
    assert len(reps) == 2 and all(o.fidelity == "screen" for o in reps)
    assert score(spec, cs, res)["replications"] == 2 and verify(res.state) == []


def test_capped_episodes_are_pending_not_scored() -> None:
    from budgeted_discovery_agent.evaluation import aggregate

    base = {"family": "f", "campaign_id": "c", "mode": "live", "status": "complete",
            "normalized_regret": 0.2, "hit": 0, "chosen_feasible": 1, "confirmed_choice": 1,
            "unused_units": 0.0}  # fmt: skip
    rows = [
        {**base, "arm": "agent", "run": 0, "stop_reason": "finished"},
        {**base, "arm": "agent", "run": 1, "stop_reason": "run_budget_exhausted",
         "status": "failed", "normalized_regret": 1.0},  # fmt: skip
    ]
    (row,) = aggregate(rows)
    assert row["episodes"] == 1 and row["pending"] == 1 and row["regret_mean"] == 0.2
