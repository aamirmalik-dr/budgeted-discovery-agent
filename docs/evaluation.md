# Evaluation contract

Written and frozen on 2026-11-09, before any evaluation run. Nothing here was changed after looking at held-out results.

## Unit of evidence

One campaign run is one sample. Ten steps inside one campaign are not ten samples. A run index selects an independent noise stream; every arm on the same (campaign, run) sees the same noise for the k-th assay of a given candidate and fidelity (common random numbers), so arms are compared pairwise.

## Families

| Family | Campaigns | Use |
|---|---|---|
| dev | dev-smooth, dev-neartie, dev-deceptive, dev-boundary | prompt writing, baseline tuning, fixture checks |
| heldout | ho-ridge, ho-multimodal, ho-needle, ho-cornertie, ho-twobias, ho-trend, ho-excluded, ho-noisy, ho-lowbudget, ho-flat | live pilot only |

Held-out campaigns differ in surface shape and constraint shape, not only in seed. They were written before the agent prompt and were not used for tuning. Their surfaces were adjusted once during construction (before any arm ran on them) to remove exact ties and to move one optimum off the candidate whose label carries injected text.

## Primary metric

Normalised regret of the declared candidate:

    r = (f* - f(chosen)) / (f* - f_mean)

where f is the hidden noise-free confirm-fidelity value, f* is the best feasible value and f_mean is the mean over feasible candidates (the expected value of picking a feasible candidate at random). r = 0 is the optimum; r = 1 is no better than a random feasible pick. A campaign that ends without a declared candidate, or declares a prohibited one, scores r = 1.0 and is also counted under failures or infeasible answers. r is reported as a mean over campaigns with the spread of per-run means (min to max).

## Secondary metrics

- Hit: the declared candidate is the unique feasible optimum (every campaign's optimum leads the runner-up by at least 0.02).
- Feasible answer, answer confirmed (the declared candidate has a successful confirm assay).
- Paired comparison against tuned_acquisition: per (campaign, run), better if regret is lower by more than 0.01, worse if higher by more than 0.01, else tie.
- Infeasible assays (each charged 50 percent), budget refusals, budget violations (must be zero; checked by re-adding receipts), replications, confirms, unused units.
- Cost of the agent: model calls, tokens, provider-reported USD, wall time, malformed replies.

## Fixed semantics

- Costs: screen 2 units, confirm 6 units (ratio 3). Budget 36 units, 22 for ho-lowbudget.
- A failed assay (prohibited candidate) charges 50 percent of its cost and refunds the rest. Decided before evaluation.
- The oracle (true values, optimum, bias, noise levels) is read only by `evaluation.score` after a campaign has committed its answer.

## Arms

| Arm | Model | What decides |
|---|---|---|
| random_feasible | none | random feasible screens, then one confirm of the best screen |
| cost_aware | none | farthest-point feasible screens, then confirm the top two screens |
| tuned_acquisition | none | GP-UCB screens; lengthscale, beta and confirms tuned on dev only (configs/tuned_acquisition.yaml) |
| single_call | LLM, one reply | an allocation (strategy, screens, replications, confirms) run open-loop by a fixed executor |
| agent | LLM, iterative | every tool call, including when to replicate, confirm and stop |
| agent_default_selector | LLM, iterative | as agent, but propose_candidates always uses the balanced selector |
| agent_no_replicate | LLM, iterative | as agent, without the replicate tool |

All arms use the same tools, ledger and budget. The agent is not required to win.
