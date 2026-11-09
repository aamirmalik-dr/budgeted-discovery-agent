from __future__ import annotations

from pathlib import Path

import pytest

from budgeted_discovery_agent.agent import make_campaign, new_state
from budgeted_discovery_agent.candidates import load_candidates
from budgeted_discovery_agent.environment import CampaignSpec, load_campaigns
from budgeted_discovery_agent.schemas import Candidate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cands() -> tuple[list[Candidate], str]:
    return load_candidates(ROOT / "tests" / "fixtures" / "candidates.json")


@pytest.fixture(scope="session")
def dev_specs() -> dict[str, CampaignSpec]:
    return {s.campaign_id: s for s in load_campaigns(ROOT / "evaluator" / "campaigns_dev.json")}


@pytest.fixture()
def boundary(cands, dev_specs):
    """A fresh campaign bundle and state on dev-boundary (has prohibited candidates)."""
    spec = dev_specs["dev-boundary"]
    cs, mhash = cands
    return spec, make_campaign(spec, cs, 0), new_state(spec, mhash, "test", 0, 50)


def prohibited_id(camp) -> str:
    return next(c for c in camp.cands if not camp.sim.feasibility(c)[0])


def allowed_id(camp) -> str:
    return next(c for c in sorted(camp.cands) if camp.sim.feasibility(c)[0])
