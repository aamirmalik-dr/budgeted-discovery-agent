# budgeted-discovery-agent

A tool-using LLM agent that runs simulated materials-screening campaigns under a hard assay budget, with every unit of spend recorded as a receipt. It decides when to explore, exploit, replicate, confirm at higher fidelity and stop, while a ledger with reservations, receipts and idempotency keys makes overspending impossible and every campaign auditable. It is evaluated on held-out synthetic campaigns against a random policy, a cost-aware design, a GP-UCB rule tuned on separate dev campaigns, and a single LLM call that fixes the whole allocation up front.

**The headline result is negative for the iterative agent.** In the first round (v1: 10 held-out campaigns, two noise streams, `openai/gpt-4.1-mini`) the tuned GP-UCB rule had the lowest regret (0.247 normalised), the single-call plan came second (0.313) and the iterative agent finished at 0.509, worse than random feasible screening (0.452), because it spent most of its budget on confirms before exploring. A second, pre-registered round (v2) on a fresh held-out family fixed a measurement bug and added one generic screening-first rule. With the same model, the single call then beat the tuned rule (0.237 against 0.300), but the iterative agent collapsed to 0.953: it screened more, then rarely stopped, and 18 of 20 episodes ran into the 80,000-token episode cap. With `anthropic/claude-sonnet-4.6` as the agent (one noise stream, 8 of 10 campaigns before the dollar cap) it reached 0.343, against 0.258 for the tuned rule on the same 8 campaigns. No arm overspent in any campaign. All campaigns and candidates are synthetic and fictional.

## Results at a glance

![Mean normalised regret by arm and round: in v1 the tuned GP-UCB rule scored 0.247, the single call 0.313 and the iterative agent 0.509; in v2 the single call scored 0.237, the tuned rule 0.300 and the agent 0.953; with the second model the agent scored 0.343 on 8 campaigns against 0.319 for the tuned rule on all 10](docs/figures/regret_by_arm.png)

In both pre-registered rounds a GP-UCB rule tuned on separate dev campaigns, or one planning call by the same model, beat the iterative gpt-4.1-mini agent: mean regret 0.247 and 0.313 against 0.509 in v1, and 0.300 and 0.237 against 0.953 in v2. In v2 the agent screened more, as the prompt fix intended, but 18 of 20 episodes hit the 80,000-token cap before it declared an answer; claude-sonnet-4.6 as the agent reached 0.343 on 8 campaigns, still behind the tuned rule on the same campaigns (0.258).

*Live, held-out families. v1 and v2: gpt-4.1-mini, 10 campaigns x 2 noise streams; second model: claude-sonnet-4.6, 10 campaigns x 1 stream, 2 agent episodes pending. Bars are means, points are individual campaigns, and the numbers on top are optimum hits with denominators.*

## What a campaign looks like

Each campaign is a hidden synthetic response surface over 66 fictional compositions on a three-component simplex (ids C-001 to C-066). The campaign has 36 assay units (22 in one held-out campaign), a screen assay at 2 units that is noisier and can be systematically biased, a confirm assay at 6 units that is accurate, and one stability rule that prohibits part of the simplex. Assaying a prohibited candidate fails and still charges half the cost. The goal is to declare the feasible candidate with the highest confirm-fidelity value.

An LLM agent runs the campaign through seven typed tools. The surrogate model and the acquisition rule are ordinary tools it can call; what the model decides is the orchestration: explore or exploit, which proposals to assay, whether a near-tie deserves a replicate, what to confirm, and when to stop. Every unit it spends goes through a ledger with reservations, receipts and idempotency keys, so the spend of any campaign can be re-added from its saved receipts.

```
python -m budgeted_discovery_agent demo
```

The fixture demo replays a scripted campaign (no model) on the dev near-tie surface and prints every decision, then the receipt ledger and its audit:

```
Receipt ledger
receipt  key  candidate  fidelity  status  reserved  charged  refunded  balance
RC-001   k3   C-034      screen    ok      2         2        0         34
...
RC-011   k20  C-054      screen    ok      2         2        0         14     <- replicate of a near-tie
RC-012   k21  C-053      screen    ok      2         2        0         12     <- replicate of a near-tie
RC-013   k22  C-054      confirm   ok      6         6        0         6
RC-014   k23  C-053      confirm   ok      6         6        0         0
Declared: C-053. Receipts sum to 36 = spent 36; remaining 0; ledger audit: consistent
```

## Not a new optimiser, and not a second active-learning repo

This repository is not a new optimisation algorithm and not a real discovery claim. The surrogate is an exact Gaussian process with a fixed RBF lengthscale and the acquisition rule is UCB, both textbook.

It also overlaps in machinery with [active-learning-microscopy](https://github.com/aamirmalik-dr/active-learning-microscopy), which uses a GP and fixed acquisition rules to decide where a simulated microscope measures next. The difference is the question. That project measures reconstruction accuracy and defect finding when every measurement costs the same, at one fidelity, with nothing prohibited. This one asks whether a language model, given the same kind of surrogate as a tool, makes better spending decisions than a tuned fixed rule when assays have two prices and two fidelities, when part of the space is prohibited, when replication is an explicit choice, and when stopping early saves budget. The GP here is simpler than the one in active-learning-microscopy (no marginal-likelihood fitting); the new parts are the cost ledger, the multi-fidelity and feasibility semantics, and the evaluation of an LLM as orchestrator. [docs/distinction.md](docs/distinction.md) has the side-by-side table.

## Results

Normalised regret is (f* - f(chosen)) / (f* - mean feasible f): 0 is the feasible optimum, 1 is no better than a random feasible pick, and a campaign with no answer scores 1. Every arm on a given (campaign, run) faces the same noise stream. The first three rows are the non-LLM baselines on the four dev campaigns, one noise stream (no model, so they are real measurements even in fixture mode). The `v1` rows are the first held-out family, 10 campaigns x 2 noise streams. The `v2` rows are a fresh held-out family (`heldout_v2`, 10 campaigns): run A repeats all arms with gpt-4.1-mini on two noise streams; run B runs the agent with claude-sonnet-4.6 against the tuned rule on one noise stream (its tuned row covers all 10 campaigns; on the 8 campaigns the Sonnet agent reached, the tuned rule scored 0.258). In every live run the non-LLM arms ran in the same pass on the same noise streams. Episodes a dollar cap prevented are shown as pending and are not in the denominators. The last two columns are model cost.

**What changed between v1 and v2, and why.** Three fixes were written down before the v2 campaigns existed, each tied to v1 evidence: the single-call executor now flags its repeat screens as replicates (all 20 v1 plans asked for them, yet the column read 0), the agent prompt gained one generic rule to screen broadly first and confirm only the final contenders (the v1 agent averaged 5.8 screens and 3.5 confirms), and the observation now lists candidates already checked and allowed (48 percent of v1 feasibility checks were repeats). Nothing was tuned after a v2 result was seen; the details are in [docs/evaluation.md](docs/evaluation.md).

<!-- measured-table:start -->
| Mode | Family | Arm | Campaigns | Normalised regret, mean (run min to max) | Hit optimum | Feasible answer | Answer confirmed | vs tuned: better / tie / worse | Infeasible assays | Budget violations | Replications | Unused units (mean) | Failed | Tokens | USD (provider) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no model | dev | random_feasible | 4 | 0.337 (0.337 to 0.337) | 0/4 | 4/4 | 4/4 | 0 / 0 / 4 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| no model | dev | cost_aware | 4 | 0.444 (0.444 to 0.444) | 0/4 | 4/4 | 4/4 | 0 / 0 / 4 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| no model | dev | tuned_acquisition | 4 | 0.000 (0.000 to 0.000) | 4/4 | 4/4 | 4/4 | reference | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| v1, no model | heldout | random_feasible | 20 | 0.452 (0.373 to 0.532) | 2/20 | 20/20 | 20/20 | 4 / 5 / 11 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| v1, no model | heldout | cost_aware | 20 | 0.546 (0.510 to 0.583) | 0/20 | 20/20 | 20/20 | 2 / 5 / 13 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| v1, no model | heldout | tuned_acquisition | 20 | 0.247 (0.187 to 0.306) | 8/20 | 20/20 | 20/20 | reference | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| v1, gpt-4.1-mini | heldout | single_call | 20 | 0.313 (0.215 to 0.411) | 9/20 | 20/20 | 19/20 | 4 / 10 / 6 | 0 | 0 | 0 | 1.5 | 0 | 32501 | 0.0120 |
| v1, gpt-4.1-mini | heldout | agent | 20 | 0.509 (0.496 to 0.523) | 4/20 | 18/20 | 18/20 | 5 / 4 / 11 | 0 | 0 | 3 | 2.0 | 2 | 1331369 | 0.5618 |
| v1, gpt-4.1-mini | heldout | agent_default_selector | 20 | 0.541 (0.502 to 0.581) | 4/20 | 15/20 | 15/20 | 4 / 4 / 12 | 0 | 0 | 1 | 4.0 | 5 | 1501084 | 0.6292 |
| v1, gpt-4.1-mini | heldout | agent_no_replicate | 20 | 0.495 (0.340 to 0.650) | 5/20 | 18/20 | 18/20 | 6 / 2 / 12 | 0 | 0 | 0 | 2.4 | 2 | 1322432 | 0.5697 |
| v2, no model | heldout_v2 | random_feasible | 20 | 0.424 (0.406 to 0.441) | 1/20 | 20/20 | 20/20 | 4 / 4 / 12 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| v2, no model | heldout_v2 | cost_aware | 20 | 0.484 (0.474 to 0.494) | 2/20 | 20/20 | 20/20 | 5 / 2 / 13 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| v2, no model | heldout_v2 | tuned_acquisition | 20 | 0.300 (0.280 to 0.319) | 7/20 | 20/20 | 20/20 | reference | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| v2, gpt-4.1-mini | heldout_v2 | single_call | 20 | 0.237 (0.144 to 0.330) | 8/20 | 20/20 | 18/20 | 5 / 11 / 4 | 0 | 0 | 60 | 0.8 | 0 | 33525 | 0.0120 |
| v2, gpt-4.1-mini | heldout_v2 | agent | 20 | 0.953 (0.900 to 1.006) | 1/20 | 2/20 | 2/20 | 3 / 0 / 17 | 0 | 0 | 0 | 12.6 | 18 | 1609932 | 0.6807 |
| v2, gpt-4.1-mini | heldout_v2 | agent_default_selector | 19 (+1 pending) | 0.912 (0.834 to 1.000) | 1/19 | 3/19 | 3/19 | 1 / 0 / 18 | 0 | 0 | 1 | 13.2 | 16 | 1508699 | 0.6134 |
| v2, gpt-4.1-mini | heldout_v2 | agent_no_replicate | 19 (+1 pending) | 0.878 (0.869 to 0.889) | 1/19 | 3/19 | 3/19 | 2 / 1 / 16 | 0 | 0 | 0 | 14.4 | 16 | 1521023 | 0.6528 |
| v2, no model | heldout_v2 | tuned_acquisition | 10 | 0.319 (0.319 to 0.319) | 4/10 | 10/10 | 10/10 | reference | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| v2, claude-sonnet-4.6 | heldout_v2 | agent | 8 (+2 pending) | 0.343 (0.343 to 0.343) | 2/8 | 6/8 | 6/8 | 2 / 1 / 5 | 0 | 0 | 0 | 8.0 | 2 | 588755 | 1.9639 |
<!-- measured-table:end -->

What the v1 pilot showed:

- **A tuned fixed rule beat every LLM arm.** GP-UCB with a lengthscale, beta and confirm count chosen on the dev campaigns reached 0.247 mean regret. Paired campaign by campaign, the agent beat it in 5 of 20, tied in 4 and lost 11; the single call beat it in 4, tied 10 and lost 6. It overfits its tuning set, though: on the dev campaigns it scored 0.000, on held-out 0.247.
- **The single call was close to the tuned rule and cheap.** Asked once for an allocation, the model chose balanced screening in all 20 campaigns, and 10 or 12 screens, three replicates and four confirms in 17 of them; the executor ran about 14 screens including the replicates and could then afford about one confirm. It found the optimum most often (9 of 20, median regret 0.067) for 0.012 USD in total.
- **The iterative agent explored too little.** It averaged 5.8 screens and 3.5 confirms per campaign, putting about 21 units, well over half its budget, into confirms, against about 14 screens and one confirm for the tuned rule. It confirmed promising-looking candidates early and then had nothing left to search with. It also asked for assays it could not afford 9 times (refused by the ledger, no charge) and ran out of tool calls before finishing in 2 of 20 campaigns.
- **The ablations change little.** Without the replicate tool the agent scored 0.495 (it replicated only 3 times in 20 campaigns anyway); with proposals forced to the balanced selector it scored 0.541 and failed to finish 5 campaigns. With 20 campaigns per arm, and the no-replicate arm's two per-run means alone spanning 0.34 to 0.65, these differences are within noise.
- **Accounting held.** Zero budget violations and zero infeasible assays in 140 live campaigns; `audit` re-adds every saved ledger. Candidate C-034 carries a label asking for ten confirms with fresh keys. It is the first space-filling proposal, so every arm assayed it. In two agent campaigns the agent confirmed C-034 four or five times; its step summaries cite C-034 being the current best and never mention the label, so the traces cannot tell whether the label contributed. Repeated confirms are also exactly the agent's general failure mode. Either way the ledger kept every campaign within budget, and one reused idempotency key was correctly charged once.
- **Cost:** 4.06 million prompt and 0.13 million completion tokens, 1.77 USD reported by the provider (1.83 USD at listed prices). Each agent arm cost about 0.56 to 0.63 USD for 20 campaigns and took about a minute per campaign.

What v2 showed:

- **The single call now beats the tuned rule.** With replicates counted properly and the screening-first rule in its prompt, the one-reply plan scored 0.237 against 0.300 for GP-UCB (paired: better in 5 of 20 campaigns, tied 11, worse 4) and found the optimum in 8 of 20. It ran 3 replicates per campaign and cost 0.012 USD for all 20.
- **The same rule broke the iterative gpt-4.1-mini agent.** Its spend moved from confirms to screens as intended (9.7 screens and 0.7 confirms per campaign, against 5.8 and 3.5 in v1), but it then kept screening and called finish in only 2 of 20 campaigns. Its episodes ran about 3 steps longer with about 9 percent longer observations, and 18 of 20 hit the 80,000-token episode cap, which scores as no answer (regret 1.0). Both ablations failed the same way. This is a negative result for the fix, reported as measured.
- **A stronger model as the agent did much better but still did not beat the tuned rule.** claude-sonnet-4.6 finished 6 of its 8 campaigns, with regret 0.343 (median 0.117) against 0.258 for GP-UCB on the same campaigns, paired better 2, tie 1, worse 5. Two episodes hit the token cap; two campaigns were not reached before the 1.85 USD re-run cap and are pending.
- **The first Sonnet attempt was an infrastructure failure, not a model result.** Every agent episode stopped on HTTP 429 rate limits while several sessions shared the provider. Its outputs are kept in `reports/aborted/live_v2_sonnet_http429/` and excluded from the table. The run was repeated once on the same frozen campaigns with only the retry settings changed (6 retries, 5 s backoff, `configs/campaign_live_v2_sonnet.yaml`); 45 rate-limit retries then recovered.
- **Accounting held again:** zero budget violations and zero infeasible assays in every v2 campaign.
- **Cost:** run A 1.96 USD reported by the provider (2.02 at listed prices; the cap is checked before each episode, so the last one ran past 2.00 and the final two LLM episodes are pending); run B 1.96 USD plus 0.31 USD for the aborted attempt.

![Median regret of the best-looking candidate as the budget is spent: at full budget the incumbent of the tuned rule reaches 0.174 (v1) and 0.113 (v2), the single call 0.067 and 0.117, and the gpt-4.1-mini agent 0.284 and 0.343, while the second-model agent reaches 0.104](docs/figures/regret_vs_spend.png)

*Live, held-out families, 20 campaign runs per arm and panel (8 for the second model). The curve is the median regret of the incumbent, the candidate with the highest mean observed value so far, against the fraction of the budget spent; before the first observation it counts as 1, and an arm that stopped early keeps its last incumbent. True values come from the frozen campaign definitions, read after the fact. In v2 the gpt-4.1-mini agent had usually found a fair candidate (median 0.343) but its declared regret is 0.953, because most episodes ended at the token cap before it declared one.*

![How LLM episodes ended: the single call always finished; the gpt-4.1-mini agent finished 18 of 20 in v1 but only 2 of 20 in v2, where 18 hit the token cap; the second-model agent finished 6 of 10, with 2 at the token cap and 2 pending](docs/figures/stop_reasons.png)

*Live, held-out families, LLM arms only (the non-LLM arms always finish). v1 and v2: gpt-4.1-mini, 20 episodes per arm; second model: claude-sonnet-4.6, 10 episodes, hatched. Pending episodes were not run because a dollar cap was reached; the aborted HTTP 429 attempt is not shown.*

![Example v2 campaign h2-negbias, run 0: the gpt-4.1-mini agent spent 31 steps, nine of them assays, reached a screen of 0.67 and a confirm of 0.62 against a true best of 0.80, and stopped at the token cap with 14 of 36 units unspent and no declared answer](docs/figures/example_campaign.png)

*Live, v2 run A, one agent episode. Chosen by a rule computed in `scripts/make_figures.py`: the lower median (5th of 10) of the run-0 agent campaigns sorted by regret, then by campaign order. Top: every assay result by kind, with ticks for proposal, feasibility and summary calls; bottom: units spent per step.*

Every failed, refused or capped episode is listed in [reports/evidence.md](reports/evidence.md); sanitized traces with the raw replies are in `reports/live*/traces/`, and each campaign's receipts are in `reports/live*/ledgers/`.

## Scorer checks (scripted replies, not model results)

In fixture mode the single-call planner and the three agent arms replay scripted replies recorded from a rule-based pilot, with planted failures listed in [tests/fixtures/README.md](tests/fixtures/README.md): a repeated idempotency key, a prose reply, a campaign that trusts the biased screen, an unchecked prohibited assay, a confirm the budget cannot pay for, three confirms ordered by the injected label, and a campaign that never finishes. These rows show that the ledger, the policy and the scorer catch what they should. They describe no model. The non-LLM rows on the dev family are real and appear in the table above.

<!-- scorer-table:start -->
| Mode | Family | Arm | Campaigns | Normalised regret, mean (run min to max) | Hit optimum | Feasible answer | Answer confirmed | vs tuned: better / tie / worse | Infeasible assays | Budget violations | Replications | Unused units (mean) | Failed | Tokens | USD (provider) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixture, scripted | dev | single_call | 4 | 0.301 (0.301 to 0.301) | 3/4 | 4/4 | 4/4 | 0 / 3 / 1 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| fixture, scripted | dev | agent | 4 | 0.444 (0.444 to 0.444) | 2/4 | 4/4 | 2/4 | 0 / 2 / 2 | 1 | 0 | 4 | 5.2 | 0 | 0 | 0 |
| fixture, scripted | dev | agent_default_selector | 4 | 0.444 (0.444 to 0.444) | 2/4 | 4/4 | 2/4 | 0 / 2 / 2 | 1 | 0 | 4 | 5.2 | 0 | 0 | 0 |
| fixture, scripted | dev | agent_no_replicate | 4 | 0.444 (0.444 to 0.444) | 2/4 | 4/4 | 2/4 | 0 / 2 / 2 | 1 | 0 | 0 | 7.2 | 0 | 0 | 0 |
| fixture, scripted | heldout | single_call | 10 | 0.330 (0.330 to 0.330) | 5/10 | 9/10 | 8/10 | 3 / 4 / 3 | 0 | 0 | 0 | 4.0 | 1 | 0 | 0 |
| fixture, scripted | heldout | agent | 10 | 0.533 (0.533 to 0.533) | 2/10 | 9/10 | 9/10 | 2 / 3 / 5 | 0 | 0 | 6 | 7.0 | 1 | 0 | 0 |
| fixture, scripted | heldout | agent_default_selector | 10 | 0.533 (0.533 to 0.533) | 2/10 | 9/10 | 9/10 | 2 / 3 / 5 | 0 | 0 | 6 | 7.0 | 1 | 0 | 0 |
| fixture, scripted | heldout | agent_no_replicate | 10 | 0.533 (0.533 to 0.533) | 2/10 | 9/10 | 9/10 | 2 / 3 / 5 | 0 | 0 | 0 | 8.2 | 1 | 0 | 0 |
<!-- scorer-table:end -->

## How it works

```mermaid
flowchart LR
    subgraph episode["Agent episode: agent arms; baselines and the single-call executor use the same tools without the loop"]
        O["Observation<br/>brief, ledger balance, observations,<br/>known feasibility, last four results"] --> M["Model adapter<br/>fixture or live"]
        M --> P["Reply parser<br/>one JSON action"]
        P --> G{"agentcore policy<br/>allowlist, argument schemas,<br/>step, tool-call and token caps"}
        G --> T["Seven tools"]
        T --> S["GP surrogate<br/>fitted on observed data only"]
        T --> L["Cost ledger<br/>reserve, commit, refund,<br/>receipts, fsynced journal"]
        T --> SIM["Simulator<br/>noisy assays, feasibility verdicts"]
        T -->|"result as untrusted data"| O
        T --> C["Checkpointed state and trace"]
    end
    H[("evaluator/<br/>hidden surfaces, bias, noise")] -.->|"read inside the simulator only"| SIM
    C -->|"declared candidate and receipts"| E["Scorer<br/>oracle read after finish,<br/>normalised regret"]
    H -.-> E
```

```
src/budgeted_discovery_agent/
  agentcore/        vendored from evidence-ledger-agent 0.1.2, unchanged: loop, policy, budgets,
                    checkpoints, adapters, sanitized traces
  candidates.py     66-candidate manifest with a SHA-256 check
  environment.py    simulator: hidden surface, screen bias, noise with common random numbers,
                    feasibility verdicts
  surrogate.py      exact GP on observed data only; explore, exploit and balanced ranking
  accounting.py     ledger: reserve, commit, refund, receipts, idempotency keys, fsynced journal
  tools.py          propose_candidates, surrogate_summary, check_feasibility, measure,
                    replicate, remaining_budget, finish
  agent.py          the iterative agent and its two ablations
  baselines.py      random feasible, cost-aware, tuned GP-UCB, single-call planner
  evaluation.py     oracle scoring after finish, aggregation, paired comparison
  report.py         evidence.json, evidence.md and the README tables
evaluator/          hidden campaign definitions (read by the simulator and the scorer only)
configs/            fixture and live configs, tuned acquisition parameters (frozen)
scripts/            campaign builders, baseline tuning on dev, fixture scripts, figures
docs/               design, evaluation contract, threat model, distinction, agentcore changes
reports/            saved outputs: ledgers, traces, task results, evidence files
```

Each step the model sees the campaign brief, its ledger balance, its observations so far, the candidates it knows to be prohibited (and, since v2, those already checked and allowed) and the last four tool results inside `trust="untrusted-data"` blocks. It replies with one JSON object naming a tool and its arguments. [docs/design.md](docs/design.md) covers the ledger semantics and checkpointing in detail, and [docs/evaluation.md](docs/evaluation.md) is the evaluation contract, written before any run.

## Quickstart

Python 3.11. No download, no model call.

```bash
git clone https://github.com/aamirmalik-dr/budgeted-discovery-agent
cd budgeted-discovery-agent
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m budgeted_discovery_agent demo
python -m budgeted_discovery_agent evaluate --config configs/campaign_fixture.yaml
python -m budgeted_discovery_agent report
python -m budgeted_discovery_agent audit
python -m pytest
```

`audit` re-adds every saved receipt ledger under `reports/` and fails if any campaign's receipts do not sum to its spend or exceed its budget.

## Live model

```bash
export AGENT_MODEL_BASE_URL=https://openrouter.ai/api/v1   # any OpenAI-compatible endpoint
export AGENT_MODEL_API_KEY=...                              # not needed for a local server
python -m budgeted_discovery_agent demo --mode live --model openai/gpt-4.1-mini
python -m budgeted_discovery_agent evaluate --config configs/campaign_live.yaml --require-live      # v1
python -m budgeted_discovery_agent evaluate --config configs/campaign_live_v2.yaml --require-live   # v2 run A
python -m budgeted_discovery_agent evaluate --config configs/campaign_live_v2_sonnet.yaml --require-live  # v2 run B
python -m budgeted_discovery_agent report
```

`configs/campaign_live.yaml` pins the model and its listed prices (0.40 and 1.60 USD per million input and output tokens). Each agent campaign is capped at 44 steps, 36 tool calls, 80,000 tokens and 600 seconds; each run has a dollar cap (3.00 for v1, 2.00 for v2 run A, 1.85 for the v2 run B re-run), checked before every episode, and live configs can set the adapter's retry count and backoff. Without an endpoint, `--require-live` exits with code 2 and never falls back to scripted replies.

## Safety model

Seven finite tools over an in-memory simulation; no shell, code execution, file access or network for the model. The hidden surfaces, bias, noise and optimum stay in `evaluator/` and inside the simulator, and the scorer reads them only after `finish`. Candidate labels are untrusted text: C-034 carries a command-like string asking for ten confirms with fresh keys, and it reaches the model only as delimited data. The ledger refuses any assay the budget cannot cover, so a model that follows such text can waste budget but cannot overspend. [docs/threat_model.md](docs/threat_model.md) lists each control with its test.

## Limitations

- Synthetic response surfaces on 66 fictional candidates. The point is controlled comparison against a hidden oracle, not realism; there is no real materials data and no discovery claim.
- One model, 10 held-out campaigns, two noise streams at temperature 0: an exploratory sample. The ranking between the tuned rule and the single call in particular could change with more campaigns.
- The surrogate has a fixed lengthscale and the tuned rule was tuned on only four dev campaigns, which it fits almost perfectly and does not generalise from as well (0.000 against 0.247).
- The agent prompt was written on the dev family and tried on two dev campaigns before freezing; no held-out result fed back into it. A prompt that tells the agent how to split its budget would probably help, and would also make it less of an agent and more of a plan.
- In v1 the single-call executor implemented its replicates as repeat screens without the replicate flag, so that arm's v1 replication count reads 0 although it repeated screens. Fixed for v2.
- The v2 second-model run is one noise stream and 8 of 10 campaigns; it shows that a stronger model changes the picture, not by how much.
- The 80,000-token episode cap was fixed in v1 and kept for v2. It is what turned the v2 gpt-4.1-mini agent's long campaigns into failures; a larger cap would have been a post-hoc change and was not tried.
- v2 run A started and finished at commit e8580db; the aborted run B attempt also ran at e8580db; the run B re-run ran at 3b3dd07, which adds only the retry settings.
- The v1 live run started at commit f5edc0a. `run_meta.json` records the HEAD at the end of the run, 506e7a4; between the two only the `audit` command changed (`git diff f5edc0a 506e7a4 -- src configs evaluator` shows one line in `__main__.py`), plus docs, tests and fixture outputs.

## Reproducibility

- Campaign definitions, the candidate manifest, configs and the tuned baseline parameters are committed and hashed into every `run_meta.json`, together with the prompt hash, commit, agentcore version, model, token counts, cost and hardware.
- Noise uses common random numbers, so every arm in a (campaign, run) pair faces the same noise, and fixture runs are deterministic.
- `scripts/tune_acquisition.py` reproduces the tuned baseline from the dev family alone (runs 100 to 104, disjoint from evaluation runs).
- `reports/evidence.json` and `reports/evidence.md` are generated from saved outputs only.
- Figures: `python scripts/make_figures.py` redraws every PNG in `docs/figures/` from the saved task results, receipt ledgers and one saved trace, plus the frozen campaign definitions for true values. It calls no model and gives identical files on every run (matplotlib comes with the `dev` extras). Each PNG is re-encoded with a 256-colour palette of its own most frequent colours to keep the repository small.

## Author

Aamir Malik

- GitHub: https://github.com/aamirmalik-dr
- LinkedIn: https://linkedin.com/in/aamirmalik-dr

## License

MIT, see [LICENSE](LICENSE). The candidates, campaigns and every other input are original and synthetic, released under the same license.
