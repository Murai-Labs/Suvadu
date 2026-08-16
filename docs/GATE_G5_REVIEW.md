# Gate G5 Review — Main Run Launch Approval

## Purpose

The always-present launch gate immediately before the single most expensive job. It exists so the cost is stated and the recipe frozen while the decision is still reversible.

## Gate Status

Status: **Not yet approved.** Approval is recorded in `docs/DECISION_LOG.md`; this file
preserves the review evidence.

## Required Evidence

| Requirement | Evidence (path) | Status |
|-------------|-----------------|--------|
| G4 recorded as PASS with human approval | `docs/GATE_G4_REVIEW.md` | TASK P4.006 |
| Projected wall-clock computed from MEASURED throughput | `docs/DECISION_LOG.md` | TASK P5.001 |
| Final recipe frozen and hashed before launch | `configs/phase5/main.yaml` | TASK P5.001 |
| Explicit dated human approval | `docs/DECISION_LOG.md` | TASK P5.001 |

Every row above is **Missing** until its task completes and its artifact is re-read at the
moment this table is filled in — not recalled from the session that produced it.

## Explicit Non-Results

- Launching before G4 passes is forbidden regardless of schedule pressure.
- A projection derived from anything other than P1.006 measured throughput is a guess and must be labelled as one.

## Required Next Work Before Approval

Complete every task named in the evidence table, then fill this table from the artifacts.

## Approval Template (paste into docs/DECISION_LOG.md when granted)

```
## DEC-XXXX — G5 approved
Date: <YYYY-MM-DD>
Task/Gate: G5
Decision: G5 (Main Run Launch Approval) approved; next phase unblocked.
Rationale: <evidence summary>.
Evidence / Source Docs: docs/GATE_G5_REVIEW.md, <run ids>.
Human Approval: <name> on <date>.
```
