# Design

## The problem being simulated

A screening campaign over 66 fictional compositions (three components, step 0.1, so a + b + c = 1). Each campaign has a hidden response surface, a cheap screen assay that is noisier and can carry a systematic bias, an expensive confirm assay, one stability rule that prohibits part of the simplex, and a hard budget in assay units. The task is to declare the best feasible candidate at confirm fidelity. The decisions that matter are what to screen, when a near-tie needs replication, what to confirm, and when to stop.

## Layers

```
evaluator/campaigns_*.json  hidden surfaces, bias, noise, constraints (read by simulator and scorer only)
environment.Simulator       noisy assays with common random numbers, feasibility verdicts
accounting.Ledger           reserve -> commit (full or 50 percent) -> refund; receipts; journal
surrogate.GPSurrogate       exact GP on observed data; explore / exploit / balanced ranking
tools.Campaign              seven typed tools over the three layers above
agent / baselines           who decides: an LLM through agentcore, fixed rules, or a one-shot plan
evaluation.score            oracle consulted after finish; normalised regret and accounting
```

The model never sees anything below the tool layer. Tools return noisy values, surrogate predictions fitted on the campaign's own observations, the feasibility verdict and its reason, and ledger balances.

## Ledger semantics

1. `measure` and `replicate` first look up the idempotency key in the state and then in the journal. A hit returns the original receipt and observation, flagged `deduplicated`, with no new charge. A key reused for a different candidate or fidelity is rejected.
2. Otherwise the ledger reserves the assay cost; if the remaining budget cannot cover it, the call is refused with `budget_exhausted` and nothing changes.
3. The simulator checks feasibility. A prohibited candidate yields a failed assay: 50 percent of the cost is charged and the rest refunded. A feasible one yields a noisy value.
4. The commit is appended to the journal and fsynced, then the state is updated: the reservation is released, spent increases by the charge, and a receipt and an observation are appended.

`verify` checks that receipts sum exactly to spent, that spent + reserved + remaining equals the budget, that no key was charged twice and that every receipt satisfies charged + refunded = reserved.

## Checkpoint and resume

The ledger lives inside `CampaignState`, so the agentcore checkpoint (written atomically after every tool result, with a write-ahead record of the pending action) carries it. The gap agentcore cannot close on its own, a crash after the assay committed but before the checkpoint was written, is closed by the journal: on resume the pending action is executed again, finds its key in the journal, and reuses the receipt.

## Surrogate

Exact GP regression, RBF kernel on the three simplex coordinates, constant mean equal to the observed mean, signal variance equal to the observed variance (floor 0.01), fixed noise of 0.10 for screen and 0.05 for confirm. Both fidelities are treated as views of one latent function, which is exactly what a biased screen violates; deciding whether to trust the screen is left to whoever calls the tool. Ties in the ranking (for example before any observation) are broken by a farthest-point order that starts at the centroid, so a cold start proposes a spread-out set. The lengthscale is 0.25 for the agent's tools; the tuned baseline uses its own tuned value.

## Arms

See docs/evaluation.md for the table. The single-call planner gets the same brief and the same tool descriptions as the agent and answers once with an allocation; a fixed executor then calls the same tools. The ablations keep the model and prompt fixed and change one tool: `agent_default_selector` makes propose ignore the requested strategy, `agent_no_replicate` removes replicate from the allowlist.

## agentcore

Vendored verbatim from evidence-ledger-agent 0.1.2 (see docs/agentcore_changes.md). The loop, policy, budgets, checkpoint store, adapters and trace writer come from there unchanged.
