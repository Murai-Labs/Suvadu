# Gate G7 Review — Memorization Audit and Publication Safety

## Purpose

Highest-precedence gate. See AGENTS.md section 11. The corpus is built from real sessions on a personal machine and contains private source code that the July 2026 scrub explicitly did not remove. Full-parameter fine-tuning memorizes; publishing the weights publishes whatever was memorized. No weights leave this machine before this gate passes - not privately, not as a draft, not temporarily.

## Gate Status

Status: **Not yet approved.** Approval is recorded in `docs/DECISION_LOG.md`; this file
preserves the review evidence.

## Required Evidence

| Requirement | Evidence (path) | Status |
|-------------|-----------------|--------|
| Extraction thresholds pre-registered BEFORE the audit runs | `docs/DECISION_LOG.md` | TASK P7.001 |
| Targeted probes across all listed private repos | `runs/phase7-extraction-001/` | TASK P7.002 |
| Prefix-prompted extraction at non-zero temperature | `runs/phase7-extraction-001/` | TASK P7.002 |
| Secret-pattern sweep over generated output | `runs/phase7-extraction-001/` | TASK P7.002 |
| Licence determination confirms redistributability | `docs/decisions/0003-dataset-licences.md` | TASK P2.003 |
| Explicit PASS/BLOCK | `this file` | TASK P7.003 |
| Model card with claim boundaries | `docs/MODEL_CARD.md` | TASK P7.004 |

Every row above is **Missing** until its task completes and its artifact is re-read at the
moment this table is filled in — not recalled from the session that produced it.

## Explicit Non-Results

- Repos that must be probed: Uyir, TamilLM, Pinnal-Core, ChittiOS (private), Kickoff-bot (Certinia/Salesforce, client-adjacent), HouseBuild, career-ops.
- On failure the remedy is corpus repair and retraining. A disclaimer in the model card is NOT an acceptable resolution.
- Results must be reported with the denominator - probes attempted - never as a bare clean.
- The --no-entropy corpus variant is LESS redacted and therefore a HIGHER memorization risk, despite being more faithful text.

## Required Next Work Before Approval

Complete every task named in the evidence table, then fill this table from the artifacts.

## Approval Template (paste into docs/DECISION_LOG.md when granted)

```
## DEC-XXXX — G7 approved
Date: <YYYY-MM-DD>
Task/Gate: G7
Decision: G7 (Memorization Audit and Publication Safety) approved; next phase unblocked.
Rationale: <evidence summary>.
Evidence / Source Docs: docs/GATE_G7_REVIEW.md, <run ids>.
Human Approval: <name> on <date>.
```
