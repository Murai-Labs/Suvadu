# Gate G4 Review — CHEAP-BASELINE FALSIFICATION (MANDATORY)

## Purpose

The standing Murai Labs gate. No phase past G4 begins until the simplest non-mechanism explanation has been RUN on metric M and FAILED to explain the effect. Two prior projects (UYIR, MARMAM) spent roughly six months combined on effects the dumbest available alternative explained just as well. Argument that a baseline is not a fair comparison is not sufficient - it must be executed and must fail.

## Gate Status

Status: **Not yet approved.** Approval is recorded in `docs/DECISION_LOG.md`; this file
preserves the review evidence.

## Required Evidence

| Requirement | Evidence (path) | Status |
|-------------|-----------------|--------|
| epsilon = 3.0 pp pre-registered before any Phase 4 run | `docs/DECISION_LOG.md DEC-0003` | Present |
| B2 prompt-only/RAG, zero training | `runs/phase4-baseline-b2-001/` | TASK P4.001 |
| B1 token-matched public-only SFT (primary null) | `runs/phase4-baseline-b1-001/` | TASK P4.002 |
| B3 traces at 10 pct subsample | `runs/phase4-baseline-b3-001/` | TASK P4.003 |
| B4 LoRA instead of full FT | `runs/phase4-baseline-b4-001/` | TASK P4.004 |
| Treatment arm at matched baseline scale | `runs/phase4-treatment-001/` | TASK P4.005 |
| Comparison table vs epsilon, with McNemar p and discordant counts | `this file` | TASK P4.006 |

Every row above is **Missing** until its task completes and its artifact is re-read at the
moment this table is filled in — not recalled from the session that produced it.

## Explicit Non-Results

- BLOCK is a legitimate and expected outcome, not a failure of execution. If a baseline explains the effect, the project re-scopes or terminates and a negative-results entry is written.
- Run B2 first - it requires no training at all. If prompting alone matches the target, the project concludes before a single GPU-hour is spent on SFT.
- A tie is a BLOCK. The treatment must exceed every baseline by more than epsilon, not merely lead.

## Required Next Work Before Approval

Complete every task named in the evidence table, then fill this table from the artifacts.

## Approval Template (paste into docs/DECISION_LOG.md when granted)

```
## DEC-XXXX — G4 approved
Date: <YYYY-MM-DD>
Task/Gate: G4
Decision: G4 (CHEAP-BASELINE FALSIFICATION (MANDATORY)) approved; next phase unblocked.
Rationale: <evidence summary>.
Evidence / Source Docs: docs/GATE_G4_REVIEW.md, <run ids>.
Human Approval: <name> on <date>.
```
