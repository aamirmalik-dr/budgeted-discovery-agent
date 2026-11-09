# budgeted-discovery-agent

RESULT_SUMMARY_PLACEHOLDER

## What a campaign looks like

Each campaign is a hidden synthetic response surface over 66 fictional compositions on a three-component simplex (ids C-001 to C-066). The campaign has 36 assay units (22 in one held-out campaign), a screen assay at 2 units that is noisier and can be systematically biased, a confirm assay at 6 units that is accurate, and one stability rule that prohibits part of the simplex. Assaying a prohibited candidate fails and still charges half the cost. The goal is to declare the feasible candidate with the highest confirm-fidelity value.

An LLM agent runs the campaign through seven typed tools. The surrogate model and the acquisition rule are ordinary tools it can call; what the model decides is the orchestration: explore or exploit, which proposals to assay, whether a near-tie deserves a replicate, what to confirm, and when to stop. Every unit it spends goes through a ledger with reservations, receipts and idempotency keys, so the spend of any campaign can be re-added from its saved receipts.

```
python -m budgeted_discovery_agent demo
```

DEMO_PLACEHOLDER

## Not a new optimiser, and not a second active-learning repo

This repository is not a new optimisation algorithm and not a real discovery claim. The surrogate is an exact Gaussian process with a fixed RBF lengthscale and the acquisition rule is UCB, both textbook.

It also overlaps in machinery with [active-learning-microscopy](https://github.com/aamirmalik-dr/active-learning-microscopy), which uses a GP and fixed acquisition rules to decide where a simulated microscope measures next. The difference is the question. That project measures reconstruction accuracy and defect finding when every measurement costs the same, at one fidelity, with nothing prohibited. This one asks whether a language model, given the same kind of surrogate as a tool, makes better spending decisions than a tuned fixed rule when assays have two prices and two fidelities, when part of the space is prohibited, when replication is an explicit choice, and when stopping early saves budget. The GP here is simpler than the one in active-learning-microscopy (no marginal-likelihood fitting); the new parts are the cost ledger, the multi-fidelity and feasibility semantics, and the evaluation of an LLM as orchestrator. [docs/distinction.md](docs/distinction.md) has the side-by-side table.

## Results

MEASURED_PLACEHOLDER

## Scorer checks (scripted replies, not model results)

In fixture mode the single-call planner and the three agent arms replay scripted replies recorded from a rule-based pilot, with planted failures listed in [tests/fixtures/README.md](tests/fixtures/README.md): a repeated idempotency key, a prose reply, a campaign that trusts the biased screen, an unchecked prohibited assay, a confirm the budget cannot pay for, three confirms ordered by the injected label, and a campaign that never finishes. These rows show that the ledger, the policy and the scorer catch what they should. They describe no model. The non-LLM rows on the dev family are real and appear in the table above.

<!-- scorer-table:start -->
<!-- scorer-table:end -->

## How it works

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
scripts/            campaign builder, baseline tuning on dev, fixture-script generator
docs/               design, evaluation contract, threat model, distinction, agentcore changes
reports/            saved outputs: ledgers, traces, task results, evidence files
```

Each step the model sees the campaign brief, its ledger balance, its observations so far, the prohibited candidates it knows about and the last four tool results inside `trust="untrusted-data"` blocks. It replies with one JSON object naming a tool and its arguments. [docs/design.md](docs/design.md) covers the ledger semantics and checkpointing in detail, and [docs/evaluation.md](docs/evaluation.md) is the evaluation contract, written before any run.

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
python -m budgeted_discovery_agent evaluate --config configs/campaign_live.yaml --require-live
python -m budgeted_discovery_agent report
```

`configs/campaign_live.yaml` pins the model and its listed prices (0.40 and 1.60 USD per million input and output tokens). Each agent campaign is capped at 44 steps, 36 tool calls, 80,000 tokens and 600 seconds; the run is capped at 3 USD, checked before every episode. Without an endpoint, `--require-live` exits with code 2 and never falls back to scripted replies.

## Safety model

Seven finite tools over an in-memory simulation; no shell, code execution, file access or network for the model. The hidden surfaces, bias, noise and optimum stay in `evaluator/` and inside the simulator, and the scorer reads them only after `finish`. Candidate labels are untrusted text: C-034 carries a command-like string asking for ten confirms with fresh keys, and it reaches the model only as delimited data. The ledger refuses any assay the budget cannot cover, so a model that follows such text can waste budget but cannot overspend. [docs/threat_model.md](docs/threat_model.md) lists each control with its test.

## Limitations

LIMITATIONS_PLACEHOLDER

## Reproducibility

- Campaign definitions, the candidate manifest, configs and the tuned baseline parameters are committed and hashed into every `run_meta.json`, together with the prompt hash, commit, agentcore version, model, token counts, cost and hardware.
- Noise uses common random numbers, so every arm in a (campaign, run) pair faces the same noise, and fixture runs are deterministic.
- `scripts/tune_acquisition.py` reproduces the tuned baseline from the dev family alone (runs 100 to 104, disjoint from evaluation runs).
- `reports/evidence.json` and `reports/evidence.md` are generated from saved outputs only.

## Author

Aamir Malik

- GitHub: https://github.com/aamirmalik-dr
- LinkedIn: https://linkedin.com/in/aamirmalik-dr

## License

MIT, see [LICENSE](LICENSE). The candidates, campaigns and every other input are original and synthetic, released under the same license.
