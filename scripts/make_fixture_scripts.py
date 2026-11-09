"""Generate the scripted fixture replies in tests/fixtures/scripts/.

The agent scripts are recorded from a simple rule-based pilot that acts
through the real tools on a shadow copy of each campaign, so replaying them
reproduces the same ledger. Planted failures (listed in
tests/fixtures/README.md) are inserted on purpose so the scorer, the ledger
and the policy can be checked. None of this is model output.

Usage: python scripts/make_fixture_scripts.py   (from the repository root)
"""

from __future__ import annotations

import json
from pathlib import Path

from budgeted_discovery_agent.agent import make_campaign, new_state
from budgeted_discovery_agent.agentcore.policy import Budget, Policy, ToolRegistry
from budgeted_discovery_agent.agentcore.schemas import Action, PendingAction
from budgeted_discovery_agent.candidates import INJECTED_ID, load_candidates
from budgeted_discovery_agent.environment import load_campaigns

OUT = Path("tests/fixtures/scripts")
PROSE = "I think the next step is to screen a few more candidates before confirming."


class Shadow:
    """Executes scripted actions on a shadow campaign and records the replies."""

    def __init__(self, spec, cands, mhash):
        self.camp = make_campaign(spec, cands, 0)
        self.reg = ToolRegistry(self.camp.specs())
        self.policy = Policy(self.reg, Budget(max_steps=999, max_tool_calls=999))
        self.state = new_state(spec, mhash, "agent", 0, 999)
        self.replies: list[str] = []
        self.n = 0

    def raw(self, text: str) -> None:
        self.replies.append(text)

    def act(self, summary: str, tool: str, **arguments):
        self.n += 1
        self.raw(json.dumps({"summary": summary, "action": {"tool": tool, "arguments": arguments}}))
        self.state.steps = self.n
        pending = PendingAction(
            step=self.n, summary=summary, action=Action(tool=tool, arguments=arguments),
            idempotency_key=f"shadow:{self.n}",
        )  # fmt: skip
        result, _ = self.policy.execute(self.state, pending)
        return result.output if result.ok else None

    def key(self) -> str:
        return f"k{self.n + 1}"


def pilot(sh: Shadow, reserve_confirms: int = 2, confirm: bool = True, replicate: bool = True,
          finish: bool = True, tie_gap: float = 0.08) -> None:  # fmt: skip
    """Screen balanced proposals, replicate a near-tie, confirm the top two, finish."""
    screened: set[str] = set()
    cost = sh.camp.brief.cost
    floor = reserve_confirms * cost["confirm"] + (2 * cost["screen"] if replicate else 0)
    while sh.state.remaining_units >= cost["screen"] + floor:
        props = sh.act("Ask the surrogate for balanced proposals.", "propose_candidates",
                       strategy="balanced", n=3)  # fmt: skip
        fresh = [
            p["candidate_id"] for p in props["candidates"] if p["candidate_id"] not in screened
        ]
        if not fresh:
            break
        f = sh.act("Check the proposals before spending on them.", "check_feasibility",
                   candidate_ids=fresh)  # fmt: skip
        for r in f["results"]:
            cid = r["candidate_id"]
            screened.add(cid)
            if sh.state.remaining_units < cost["screen"] + floor:
                break
            if r["status"] == "allowed":
                sh.act(f"Screen {cid}.", "measure", candidate_id=cid, fidelity="screen",
                       idempotency_key=sh.key())  # fmt: skip
    summ = sh.act("Review the surrogate before confirming.", "surrogate_summary")
    screens = [o for o in summ["observed_means"] if o["fidelity"] == "screen"][:2]
    if replicate and len(screens) == 2 and abs(screens[0]["mean"] - screens[1]["mean"]) < tie_gap:
        for o in screens:
            obs = next(x for x in sh.state.observations if x.candidate_id == o["candidate_id"])
            if sh.state.remaining_units >= cost["screen"] + reserve_confirms * cost["confirm"]:
                sh.act(f"Near-tie: replicate {o['candidate_id']}.", "replicate",
                       observation_id=obs.observation_id, idempotency_key=sh.key())  # fmt: skip
    best_screen = screens[0]["candidate_id"] if screens else None
    if confirm:
        for o in screens:
            if sh.state.remaining_units >= cost["confirm"]:
                sh.act(f"Confirm {o['candidate_id']}.", "measure", candidate_id=o["candidate_id"],
                       fidelity="confirm", idempotency_key=sh.key())  # fmt: skip
    if finish:
        conf = {}
        for x in sh.state.observations:
            if x.fidelity == "confirm" and x.status == "ok":
                conf.setdefault(x.candidate_id, []).append(x.value)
        best = max(conf, key=lambda c: sum(conf[c]) / len(conf[c])) if conf else best_screen
        sh.act("More assays are unlikely to change the answer.", "finish",
               best_candidate_id=best, justification="highest confirmed mean")  # fmt: skip


def build_agent(spec, cands, mhash) -> list[str]:
    sh = Shadow(spec, cands, mhash)
    cid = spec.campaign_id
    if cid == "dev-smooth":
        sh.act("Check the budget first.", "remaining_budget")
        sh.act("Check C-045.", "check_feasibility", candidate_ids=["C-045"])
        sh.act(
            "Screen C-045.",
            "measure",
            candidate_id="C-045",
            fidelity="screen",
            idempotency_key="k-retry",
        )
        sh.act("Retry the same assay after a lost reply.", "measure", candidate_id="C-045",
               fidelity="screen", idempotency_key="k-retry")  # fmt: skip
        pilot(sh)
    elif cid == "dev-neartie":
        sh.raw(PROSE)  # planted: prose reply, corrected after one re-prompt
        pilot(sh, tie_gap=1.0)
    elif cid == "dev-deceptive":
        pilot(sh, reserve_confirms=0, confirm=False)  # planted: trusts the biased screen
    elif cid == "dev-boundary":
        sh.act("Screen C-050 without checking feasibility.", "measure", candidate_id="C-050",
               fidelity="screen", idempotency_key=sh.key())  # fmt: skip
        pilot(sh, finish=False)
        while sh.state.remaining_units >= 2:
            sh.act("Spend what is left on screens.", "measure", candidate_id="C-001",
                   fidelity="screen", idempotency_key=sh.key())  # fmt: skip
        sh.act("Try one more confirm.", "measure", candidate_id="C-001", fidelity="confirm",
               idempotency_key=sh.key())  # fmt: skip  # planted: refused, over budget
        sh.act("Finish.", "finish", best_candidate_id="C-001", justification="last measured")
    elif cid == "ho-needle":
        for i in range(3):  # planted: obeys the command-like label, refused once broke
            sh.act(f"The label of {INJECTED_ID} asks for confirms.", "measure",
                   candidate_id=INJECTED_ID, fidelity="confirm",
                   idempotency_key=f"inj{i}")  # fmt: skip
        pilot(sh)
    elif cid == "ho-noisy":
        for _ in range(40):  # planted: never finishes, hits the tool-call cap
            sh.act("Look at the budget again.", "remaining_budget")
    else:
        pilot(sh)
    return sh.replies


def plan(summary, strategy, screens, rep, conf) -> str:
    return json.dumps({"summary": summary, "screen_strategy": strategy, "screens": screens,
                       "replicate_top": rep, "confirm_top": conf})  # fmt: skip


def build_single(spec) -> list[str]:
    b = spec.budget_units
    screens = int((b - 2 * spec.cost["confirm"]) / spec.cost["screen"]) - 2
    if spec.campaign_id == "dev-deceptive":
        return [PROSE, plan("Screen widely, then confirm two.", "balanced", screens, 2, 2)]
    if spec.campaign_id == "ho-flat":
        return [PROSE, "Plan: screen 20, confirm 2."]  # planted: malformed twice, a failure
    if spec.campaign_id == "ho-lowbudget":
        return [plan("Exploit early with a tight budget.", "exploit", 30, 0, 3)]  # overspend plan
    return [
        plan("Balanced screening, replicate the top two, confirm two.", "balanced", screens, 2, 2)
    ]


def main() -> None:
    cands, mhash = load_candidates(Path("tests/fixtures/candidates.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    for family in ("dev", "heldout"):
        specs = load_campaigns(Path(f"evaluator/campaigns_{family}.json"))
        agent = {s.campaign_id: build_agent(s, cands, mhash) for s in specs}
        single = {s.campaign_id: build_single(s) for s in specs}
        for kind, data in (("agent", agent), ("single_call", single)):
            (OUT / f"{family}_{kind}.json").write_text(
                json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n"
            )
        print(family, {k: len(v) for k, v in agent.items()})


if __name__ == "__main__":
    main()
