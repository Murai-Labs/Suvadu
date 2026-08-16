# Gate G3 Review — Evaluation Frozen and Base Measured

## Purpose

G3 fixes the metric before any arm is trained, and produces the base reference row. Without a base measurement no training result is interpretable, and without freezing the metric first the suite can be chosen after the fact to flatter the outcome.

## Gate Status

Status: **Not yet approved.** Approval is recorded in `docs/DECISION_LOG.md`; this file
preserves the review evidence.

## Required Evidence

| Requirement | Evidence (path) | Status |
|-------------|-----------------|--------|
| Eval suites pinned with revision SHAs | `configs/phase3/evals.yaml` | TASK P3.001 |
| Paired McNemar harness ported and reproducing a known result | `src/suvadu/eval/compare.py` | TASK P3.002 |
| Base Qwen3.8-27B scored, per-item results persisted | `runs/phase3-base-001/` | TASK P3.003 |
| Power calculation fixing n | `docs/DECISION_LOG.md` | TASK P3.004 |
| Metric, harness, suites, n and epsilon frozen together | `this file` | TASK P3.005 |

Every row above is **Missing** until its task completes and its artifact is re-read at the
moment this table is filled in — not recalled from the session that produced it.

## Explicit Non-Results

- No fine-tuned model exists yet.
- Per-item results are mandatory - totals alone cannot be paired, and an unpaired comparison at n=200 tells you nothing.
- Before any aggregate is reported, at least three individual items must be read to confirm the scorer works.

## Required Next Work Before Approval

Complete every task named in the evidence table, then fill this table from the artifacts.

## Approval Template (paste into docs/DECISION_LOG.md when granted)

```
## DEC-XXXX — G3 approved
Date: <YYYY-MM-DD>
Task/Gate: G3
Decision: G3 (Evaluation Frozen and Base Measured) approved; next phase unblocked.
Rationale: <evidence summary>.
Evidence / Source Docs: docs/GATE_G3_REVIEW.md, <run ids>.
Human Approval: <name> on <date>.
```
