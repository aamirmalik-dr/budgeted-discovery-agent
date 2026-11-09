# Threat model

## What the model can do

Seven finite, typed tools over an in-memory simulation: propose_candidates, surrogate_summary, check_feasibility, measure, replicate, remaining_budget, finish. No shell, no code execution, no file writes, no network. The only side effects are ledger entries inside the simulation and, for the agent arms, an append-only journal file the harness writes (the model never names a path).

## What is protected, and how

| Risk | Control | Test |
|---|---|---|
| Overspending the budget | the ledger refuses any reservation larger than the remaining units; the refusal is a typed `budget_exhausted` error | `test_insufficient_funds_is_refused_without_any_charge`, `test_planted_over_budget_attempt_is_refused_and_episode_continues` |
| Double charge on a retry | idempotency keys: a repeated key returns the original receipt; a key reused for a different assay is rejected | `test_duplicate_idempotency_key_charges_once` |
| Double charge after a crash | every commit is fsynced to a journal before the tool returns; resume merges journalled receipts | `test_journal_lets_resume_reuse_a_committed_receipt`, `test_resume_does_not_repeat_a_committed_measurement` |
| Ledger tampering or arithmetic drift | `verify` re-adds receipts against spent, reserved and remaining; `audit` re-adds every saved ledger | `test_verify_detects_tampered_ledger` |
| Oracle leakage | true values, optimum, screen bias and noise live in `evaluator/` and inside the simulator; no tool returns them; scoring runs after `finish` | `test_tools_expose_no_oracle` |
| Instructions inside candidate labels | labels are returned only inside delimited `trust="untrusted-data"` result blocks; the allowlist and the budget are fixed outside the model. Candidate C-034 carries a command-like label | `test_command_like_label_stays_data` |
| Malformed or oversized replies | agentcore parser with one corrective re-prompt; a second failure costs a step | `test_malformed_reply_and_adapter_failure_are_recorded` |
| Runaway episodes | caps on steps, tool calls, tokens and wall time per campaign, and on tokens and dollars per evaluation run | configs |

## What is not protected

The ledger prevents overspending, not wasteful spending. A model that obeys the injected label can still burn its budget on pointless confirms within the limit; that shows up as regret, not as a violation. The journal protects against a crash between commit and checkpoint, not against a corrupted journal file. Traces are sanitized for keys and absolute paths but contain raw model replies.
