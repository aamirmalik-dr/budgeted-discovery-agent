# Evidence

Generated from saved outputs by `python -m budgeted_discovery_agent report`. Do not edit by hand.

## Fixture mode

Fixture mode replays scripted replies for single_call and the three agent arms, with planted failures. Those rows check the ledger, policy and scorer; they are not model results. The non-LLM baselines call no model, so their fixture rows are real measurements on one noise stream per campaign.

- Commit: `f5edc0a663ff170bf67d244a42a21cce34072450` (working tree dirty: False)
- Model: none (scripted fixture adapter); agentcore 0.1.2
- Prompt: campaign-prompts-v1, sha256 `2006e7b731e7152d`
- Config sha256: `c1ff79546097b700`; tuned acquisition: {'beta': 0.5, 'lengthscale': 0.15, 'n_confirm': 1}
- Run: 2026-11-09T12:35:51+09:00 to 2026-11-09T12:40:17+09:00
- Hardware: Windows-10-10.0.26300-SP0, Python 3.11.9
- Campaign episodes scored: 98
- Tokens: None, USD at listed prices: None, USD reported by provider: None, adapter retries: None

### Input hashes

- candidates: `7b332d9ee44be070`
- dev/campaigns: `eaca1ebfd7678eb7`
- dev/script/agent: `286a04d1c030fcec`
- dev/script/single_call: `53e32bd9da00831e`
- heldout/campaigns: `53b8c83268bec834`
- heldout/script/agent: `36935ad34e06f596`
- heldout/script/single_call: `4f8712d8cd95f6fc`
- tuned_params: `f8a623418795df2c`

### Every failed, infeasible, refused or malformed episode

| Family | Arm | Campaign | Run | Status | Stop | Chosen | True best | Regret | Infeasible | Refused | Malformed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | agent | dev-boundary | 0 | complete | finished | C-001 | C-044 | 1.1393 | 1 | 1 | 0 |
| dev | agent_default_selector | dev-boundary | 0 | complete | finished | C-001 | C-044 | 1.1393 | 1 | 1 | 0 |
| dev | agent_no_replicate | dev-boundary | 0 | complete | finished | C-001 | C-044 | 1.1393 | 1 | 1 | 0 |
| heldout | agent | ho-noisy | 0 | stopped | max_tool_calls | None | C-043 | 1.0 | 0 | 0 | 0 |
| heldout | agent_default_selector | ho-noisy | 0 | stopped | max_tool_calls | None | C-043 | 1.0 | 0 | 0 | 0 |
| heldout | agent_no_replicate | ho-noisy | 0 | stopped | max_tool_calls | None | C-043 | 1.0 | 0 | 0 | 0 |
| heldout | single_call | ho-flat | 0 | failed | malformed_output | None | C-027 | 1.0 | 0 | 0 | 1 |

## Live mode

Live model results on the held-out family; non-LLM rows share the same noise streams.

- Commit: `506e7a4079b70299e067fd2e7ae132230c7c4bed` (working tree dirty: False)
- Model: openai/gpt-4.1-mini; agentcore 0.1.2
- Prompt: campaign-prompts-v1, sha256 `2006e7b731e7152d`
- Config sha256: `546c35d5c0e84095`; tuned acquisition: {'beta': 0.5, 'lengthscale': 0.15, 'n_confirm': 1}
- Run: 2026-11-09T12:35:53+09:00 to 2026-11-09T13:35:26+09:00
- Hardware: Windows-10-10.0.26300-SP0, Python 3.11.9
- Campaign episodes scored: 140
- Tokens: {'completion': 127028, 'prompt': 4060358}, USD at listed prices: 1.8273880000000002, USD reported by provider: 1.77267, adapter retries: 0

### Input hashes

- candidates: `7b332d9ee44be070`
- heldout/campaigns: `53b8c83268bec834`
- tuned_params: `f8a623418795df2c`

### Every failed, infeasible, refused or malformed episode

| Family | Arm | Campaign | Run | Status | Stop | Chosen | True best | Regret | Infeasible | Refused | Malformed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| heldout | agent_no_replicate | ho-ridge | 0 | complete | finished | C-032 | C-023 | 0.0828 | 0 | 1 | 0 |
| heldout | agent_default_selector | ho-needle | 0 | stopped | max_tool_calls | None | C-027 | 1.0 | 0 | 0 | 0 |
| heldout | agent | ho-cornertie | 0 | complete | finished | C-064 | C-064 | 0.0 | 0 | 1 | 0 |
| heldout | agent_no_replicate | ho-twobias | 0 | complete | finished | C-019 | C-018 | 0.4054 | 0 | 1 | 0 |
| heldout | agent | ho-excluded | 0 | stopped | max_tool_calls | None | C-020 | 1.0 | 0 | 3 | 0 |
| heldout | agent | ho-noisy | 0 | complete | finished | C-049 | C-043 | 0.2835 | 0 | 3 | 0 |
| heldout | agent_default_selector | ho-noisy | 0 | stopped | max_tokens | None | C-043 | 1.0 | 0 | 0 | 0 |
| heldout | agent_no_replicate | ho-lowbudget | 0 | complete | finished | C-005 | C-057 | 0.6025 | 0 | 1 | 0 |
| heldout | agent_default_selector | ho-flat | 0 | stopped | max_tool_calls | None | C-027 | 1.0 | 0 | 0 | 0 |
| heldout | agent_no_replicate | ho-flat | 0 | complete | finished | C-036 | C-027 | 0.4234 | 0 | 1 | 0 |
| heldout | agent_default_selector | ho-ridge | 1 | stopped | max_tool_calls | None | C-023 | 1.0 | 0 | 0 | 0 |
| heldout | agent_no_replicate | ho-multimodal | 1 | stopped | max_tool_calls | None | C-020 | 1.0 | 0 | 0 | 0 |
| heldout | agent_default_selector | ho-needle | 1 | complete | finished | C-049 | C-027 | 0.8242 | 0 | 2 | 0 |
| heldout | agent_no_replicate | ho-needle | 1 | stopped | max_tool_calls | None | C-027 | 1.0 | 0 | 0 | 0 |
| heldout | agent | ho-twobias | 1 | complete | finished | C-008 | C-018 | 0.4054 | 0 | 1 | 0 |
| heldout | agent_no_replicate | ho-trend | 1 | complete | finished | C-023 | C-022 | 0.2398 | 0 | 1 | 0 |
| heldout | agent_default_selector | ho-excluded | 1 | complete | finished | C-009 | C-020 | 0.4091 | 0 | 2 | 0 |
| heldout | agent_no_replicate | ho-excluded | 1 | complete | finished | C-020 | C-020 | 0.0 | 0 | 1 | 0 |
| heldout | agent_default_selector | ho-noisy | 1 | stopped | max_tool_calls | None | C-043 | 1.0 | 0 | 0 | 0 |
| heldout | agent | ho-lowbudget | 1 | complete | finished | C-034 | C-057 | 1.1882 | 0 | 1 | 0 |
| heldout | agent_default_selector | ho-lowbudget | 1 | complete | finished | C-057 | C-057 | 0.0 | 0 | 3 | 0 |
| heldout | agent | ho-flat | 1 | stopped | max_tool_calls | None | C-027 | 1.0 | 0 | 0 | 0 |
| heldout | agent_no_replicate | ho-flat | 1 | complete | finished | C-010 | C-027 | 1.2989 | 0 | 1 | 0 |

## Measured results

| Mode | Family | Arm | Campaigns | Normalised regret, mean (run min to max) | Hit optimum | Feasible answer | Answer confirmed | vs tuned: better / tie / worse | Infeasible assays | Budget violations | Replications | Unused units (mean) | Failed | Tokens | USD (provider) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no model | dev | random_feasible | 4 | 0.337 (0.337 to 0.337) | 0/4 | 4/4 | 4/4 | 0 / 0 / 4 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| no model | dev | cost_aware | 4 | 0.444 (0.444 to 0.444) | 0/4 | 4/4 | 4/4 | 0 / 0 / 4 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| no model | dev | tuned_acquisition | 4 | 0.000 (0.000 to 0.000) | 4/4 | 4/4 | 4/4 | reference | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| no model | heldout | random_feasible | 20 | 0.452 (0.373 to 0.532) | 2/20 | 20/20 | 20/20 | 4 / 5 / 11 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| no model | heldout | cost_aware | 20 | 0.546 (0.510 to 0.583) | 0/20 | 20/20 | 20/20 | 2 / 5 / 13 | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| no model | heldout | tuned_acquisition | 20 | 0.247 (0.187 to 0.306) | 8/20 | 20/20 | 20/20 | reference | 0 | 0 | 0 | 0.0 | 0 | 0 | 0 |
| live | heldout | single_call | 20 | 0.313 (0.215 to 0.411) | 9/20 | 20/20 | 19/20 | 4 / 10 / 6 | 0 | 0 | 0 | 1.5 | 0 | 32501 | 0.0120 |
| live | heldout | agent | 20 | 0.509 (0.496 to 0.523) | 4/20 | 18/20 | 18/20 | 5 / 4 / 11 | 0 | 0 | 3 | 2.0 | 2 | 1331369 | 0.5618 |
| live | heldout | agent_default_selector | 20 | 0.541 (0.502 to 0.581) | 4/20 | 15/20 | 15/20 | 4 / 4 / 12 | 0 | 0 | 1 | 4.0 | 5 | 1501084 | 0.6292 |
| live | heldout | agent_no_replicate | 20 | 0.495 (0.340 to 0.650) | 5/20 | 18/20 | 18/20 | 6 / 2 / 12 | 0 | 0 | 0 | 2.4 | 2 | 1322432 | 0.5697 |

## Scorer checks (scripted replies, not model results)

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

## Limitations

- Synthetic response surfaces on a 66-candidate fictional simplex; no real materials data and no discovery claim.
- The surrogate is a fixed-hyperparameter GP and the acquisition rule is textbook UCB; this is not a new optimisation algorithm.
- Normalised regret uses the mean feasible value as the zero-skill reference; a campaign with no answer or an infeasible answer scores 1.0.
- Fixture results for LLM arms are scripted and say nothing about model quality.
- The live pilot is one model, 10 held-out campaigns and two runs at temperature 0: an exploratory sample with min and max over runs, not a ranking.

## Reproduce

- fixture: `python -m budgeted_discovery_agent evaluate --config configs/campaign_fixture.yaml`
- live: `python -m budgeted_discovery_agent evaluate --config configs/campaign_live.yaml --require-live`
- report: `python -m budgeted_discovery_agent report`
- audit: `python -m budgeted_discovery_agent audit`
