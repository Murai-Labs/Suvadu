# Gate G6 Review — Capability Gate

## Purpose

G6 tests the claim that justifies publication: that the fine-tuned model beats base Qwen3.8-27B at tool calling and coding. It also re-checks the G4 result at full scale, because the shipped model is trained differently from the baseline-scale arms.

## Gate Status

Status: **Not yet approved.** Approval is recorded in `docs/DECISION_LOG.md`; this file
preserves the review evidence.

## Required Evidence

| Requirement | Evidence (path) | Status |
|-------------|-----------------|--------|
| Shipping checkpoint selected on per-epoch evidence | `docs/decisions/0006-checkpoint-selection.md` | TASK P5.003 |
| Capability retention outside the training domain measured | `runs/phase5-retention-001/` | TASK P5.004 |
| Paired comparison vs base, p and discordant counts | `runs/phase6-capability-001/` | TASK P6.001 |
| Traces advantage over B1 re-confirmed at scale | `this file` | TASK P6.002 |
| Explicit PASS/BLOCK | `this file` | TASK P6.003 |

Every row above is **Missing** until its task completes and its artifact is re-read at the
moment this table is filled in — not recalled from the session that produced it.

## Explicit Non-Results

- PASS requires a win on tool calling AND coding that exceeds the 6-discordant-item sensitivity floor. A win on one and a tie on the other is not a PASS.
- Retention regressions are disclosed in the decision and the model card, not deferred.
- A PASS on the paired test is not proof of equivalence elsewhere - it means only that this test at this n detected a difference.

## Required Next Work Before Approval

Complete every task named in the evidence table, then fill this table from the artifacts.

## Approval Template (paste into docs/DECISION_LOG.md when granted)

```
## DEC-XXXX — G6 approved
Date: <YYYY-MM-DD>
Task/Gate: G6
Decision: G6 (Capability Gate) approved; next phase unblocked.
Rationale: <evidence summary>.
Evidence / Source Docs: docs/GATE_G6_REVIEW.md, <run ids>.
Human Approval: <name> on <date>.
```
