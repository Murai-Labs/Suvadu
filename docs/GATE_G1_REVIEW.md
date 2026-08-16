# Gate G1 Review — Training Stack Verified

## Purpose

G1 proves a 27B full-parameter optimizer step is physically possible on this cluster BEFORE days are spent building a corpus. If the toolchain does not exist for aarch64 + sm_121, the project re-scopes to LoRA, and that must be discovered now rather than after the data work.

## Gate Status

Status: **Not yet approved.** Approval is recorded in `docs/DECISION_LOG.md`; this file
preserves the review evidence.

## Required Evidence

| Requirement | Evidence (path) | Status |
|-------------|-----------------|--------|
| Toolchain report, transformers v5 + sm_121 confirmed | `configs/phase1/env-report.md` | TASK P1.001 |
| BF16 base resident, revision pinned | `configs/phase1/base-model.json` | TASK P1.002 |
| Measured memory profile, full FT and LoRA | `runs/phase1-memprobe-001/metrics.json` | TASK P1.003 |
| Training entrypoint with progress + provenance | `src/suvadu/cli/train.py` | TASK P1.004 |
| DeepSeek restore procedure captured BEFORE shutdown | `docs/RUNBOOK.md restore section` | TASK P1.005 |
| Smoke run >=50 steps, loss decreasing | `runs/phase1-smoke-001/` | TASK P1.006 |
| Measured throughput and projected main-run wall-clock | `runs/phase1-smoke-001/metrics.json` | TASK P1.006 |

Every row above is **Missing** until its task completes and its artifact is re-read at the
moment this table is filled in — not recalled from the session that produced it.

## Explicit Non-Results

- No corpus exists.
- No evaluation has been run.
- The 162 GB estimate remains arithmetic until P1.003 measures it.
- Whether full-parameter FT fits across 2 nodes is UNKNOWN, not assumed.

## Required Next Work Before Approval

Complete every task named in the evidence table, then fill this table from the artifacts.

## Approval Template (paste into docs/DECISION_LOG.md when granted)

```
## DEC-XXXX — G1 approved
Date: <YYYY-MM-DD>
Task/Gate: G1
Decision: G1 (Training Stack Verified) approved; next phase unblocked.
Rationale: <evidence summary>.
Evidence / Source Docs: docs/GATE_G1_REVIEW.md, <run ids>.
Human Approval: <name> on <date>.
```
