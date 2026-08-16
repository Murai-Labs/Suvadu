# Gate G2 Review — Data Frozen

## Purpose

G2 freezes every arm's corpus hash. After this gate the data is immutable: re-exporting mid-experiment would silently change the inputs of runs already completed, making arms incomparable. It also carries the two safety checks that publication depends on - decontamination and licence review.

## Gate Status

Status: **Not yet approved.** Approval is recorded in `docs/DECISION_LOG.md`; this file
preserves the review evidence.

## Required Evidence

| Requirement | Evidence (path) | Status |
|-------------|-----------------|--------|
| Traces re-exported to current date with counts | `configs/phase2/traces-export.json` | TASK P2.001 |
| Private-source-code audit, hand-read sample | `docs/decisions/0002-corpus-safety.md` | TASK P2.002 |
| Licence determination, every dataset | `docs/decisions/0003-dataset-licences.md` | TASK P2.003 |
| Merge pipeline, chat template verified | `src/suvadu/data/build_corpus.py` | TASK P2.004 |
| Reasoning-token format decision | `docs/decisions/0004-thinking-tokens.md` | TASK P2.005 |
| Decontamination report, zero residual overlap | `configs/phase2/decontam-report.json` | TASK P2.006 |
| Token-matched control corpus within 2 pct | `configs/phase2/mixture-control.yaml` | TASK P2.007 |
| All corpus hashes recorded and frozen | `this file` | TASK P2.008 |

Every row above is **Missing** until its task completes and its artifact is re-read at the
moment this table is filled in — not recalled from the session that produced it.

## Explicit Non-Results

- No training has been run on this corpus.
- Decontamination removals must be hand-checked; a zero-overlap number alone is not evidence the sweep was correct.
- Licence status of the r0b0tlab datasets was UNRESOLVED as of 2026-08-16 - all seven displayed no licence.

## Required Next Work Before Approval

Complete every task named in the evidence table, then fill this table from the artifacts.

## Approval Template (paste into docs/DECISION_LOG.md when granted)

```
## DEC-XXXX — G2 approved
Date: <YYYY-MM-DD>
Task/Gate: G2
Decision: G2 (Data Frozen) approved; next phase unblocked.
Rationale: <evidence summary>.
Evidence / Source Docs: docs/GATE_G2_REVIEW.md, <run ids>.
Human Approval: <name> on <date>.
```
