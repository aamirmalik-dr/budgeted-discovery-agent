# How this differs from existing repositories

Two repositories in the same portfolio already choose measurements with a surrogate. This one should not be read as a third version of them.

## active-learning-microscopy

That project is a simulation study of where an autonomous microscope should measure next on a 64 x 64 synthetic field. Its goal is reconstruction accuracy (map RMSE against measurement count) and rare-defect finding. Every measurement costs the same, there is one fidelity, nothing is infeasible, and the question of when to stop does not arise beyond a fixed budget curve. The decision rule is a fixed acquisition function (GP variance, gradient-weighted variance, expected exceedance), and no language model is involved.

## forcefield-active-learning

That project selects which configurations to label for a machine-learned interatomic potential, by ensemble disagreement. Again one label cost, one fidelity, a fixed rule and no language model.

## This repository

| | active-learning-microscopy | this repository |
|---|---|---|
| Goal | reconstruct a field, find defects | declare the single best feasible candidate |
| Candidates | 4096 grid positions | 66 fictional simplex compositions |
| Cost | uniform | screen 2 units, confirm 6 units, hard budget |
| Fidelities | one | two; screen can be biased |
| Feasibility | none | prohibited region, failed assays charge 50 percent |
| Replication | not a decision | an explicit tool |
| Accounting | measurement count | reservation, commit, refund, receipts, idempotency keys, journal |
| Who decides | fixed acquisition rule | an LLM choosing tools, compared against a tuned fixed rule |

## What overlaps, plainly

The surrogate here is an exact GP with an RBF kernel and the acquisition rule is UCB. Both are textbook and are conceptually the same machinery that active-learning-microscopy implements more carefully (that repository fits hyperparameters by marginal likelihood; this one uses a fixed lengthscale chosen on dev campaigns). Nothing in this repository is a better optimiser. The new parts are the budget ledger, the multi-fidelity and feasibility semantics, and the evaluation of an LLM as the orchestrator of ordinary tools against that tuned fixed rule.
