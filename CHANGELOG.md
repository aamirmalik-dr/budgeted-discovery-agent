# Changelog

All live results were produced through OpenRouter at temperature 0. Values are mean normalised regret of the declared candidate on held-out campaigns. Lower is better, and a campaign with no answer scores 1. Every version keeps the full history of saved outputs, ledgers and traces.

## 0.2.0 (2026-11-10)

v2 round, pre-registered before the v2 campaigns existed (see `docs/evaluation.md`). Three changes:
- The single-call executor flags its repeat screens as replicates.
- The agent prompt adds a screening-first budget rule.
- The observation lists candidates already checked and allowed.

Campaigns:
- New constraint kinds: two-disk, wedge and ring.
- A frozen held-out v2 family.
- Episodes stopped by a dollar cap are reported as pending.

Live results with `openai/gpt-4.1-mini` (10 campaigns x 2 noise streams):

| Arm | Regret |
|---|---|
| single call | 0.237 |
| tuned GP-UCB | 0.300 |
| agent | 0.953 |

The agent hit the 80,000-token cap in 18 of 20 episodes.

Live results with `anthropic/claude-sonnet-4.6`, 1 stream:
- The agent scored 0.343 on 8 campaigns. 2 were pending at the dollar cap.
- The tuned rule scored 0.258 on the same 8 campaigns.
- The first attempt hit HTTP 429 rate limits. It is kept under `reports/aborted/` and not counted.
- Live configs take the adapter retry count and backoff.
- Saved model replies are written as ASCII-escaped JSON.

Figures:
- `scripts/make_figures.py` draws every figure from saved task results, receipt ledgers and one trace, with true values read from the frozen campaign definitions.
- The current set:
  - Figure 1: incumbent regret against budget spent (small multiples with bootstrap bands), final-regret rainclouds, stop reasons, and an example campaign raster.
  - Figure S1: paired difference against the tuned rule.
  - Exported as SVG and as 256-colour 300 dpi PNG.
  - Campaign-level bootstrap intervals.
  - Every plotted value is in `docs/figures/figure_stats.json`.
  - The typeface, Liberation Sans, is vendored under the SIL Open Font License.
- README:
  - "Main result" section, Statistical methods paragraph and a Mermaid architecture diagram.
  - One text correction: the observation also lists checked candidates.
  - No result number changed.
- `tests/test_figures.py` checks that every figure is produced, stays within size limits and is reproducible.

## 0.1.0 (2026-11-09)

- agentcore 0.1.2, vendored from evidence-ledger-agent.
- Components:
  - a candidate manifest;
  - a deterministic simulator with common random numbers across arms;
  - a GP surrogate;
  - a cost ledger with reservations, receipts and idempotency keys;
  - campaign tools.
- Arms:
  - the agent and two ablations (no replicate, default selector);
  - random feasible;
  - cost-aware design;
  - GP-UCB tuned on dev campaigns;
  - a single planning call.
- Evaluation, reports, CLI and ledger audit, with fixture scripts and tests.
- v1 live pilot with `openai/gpt-4.1-mini` (10 campaigns x 2 noise streams):

| Arm | Regret |
|---|---|
| tuned GP-UCB | 0.247 |
| single call | 0.313 |
| random feasible | 0.452 |
| agent | 0.509 |

- Zero budget violations and zero infeasible assays.
