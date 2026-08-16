# Suvadu — Experiment Log

## Update Rules

Append an entry after EVERY run — including failed, skipped, or inconclusive runs. Never
overwrite. Run IDs are descriptive and never reused.

## Entry Template

```
Run ID:            <phaseN-purpose-NNN>
Task ID:           <TASK id from atomic list>
Date:              <YYYY-MM-DD>
Git Commit:        <SHA>
Git Status:        <clean / dirty (list)>
Exact Command:     <command>
Config Path:       <configs/...>
Config Hash:       <sha256>
Data Hash:         <sha256 / split id>
Seed:              <int>
Environment:       <GPU, CUDA, Python, key package versions>
Also Running:      <what else occupied the cluster during this run>
Checkpoint Path:   <path or n/a>
Metrics Path:      <runs/.../metrics.json>
Status:            <success / failed / inconclusive>
Failure Notes:     <if applicable>
Interpretation:    <what it means; claim boundary>
Next Action:       <follow-up>
```

## Run ID Allocation

Descriptive IDs preserving phase and purpose:
`phase1-memprobe-001`, `phase3-base-001`, `phase4-baseline-b1-001`, `phase4-treatment-001`,
`phase5-main-001`, `phase6-capability-001`, `phase7-extraction-001`.

Do not reuse IDs, even for failed attempts. A failed `phase5-main-001` is followed by
`phase5-main-002`, and the failure entry stays.

---

## Runs

**No runs have been executed.** This project has produced zero measurements as of 2026-08-16.

The first entry will be `phase1-memprobe-001` (TASK P1.003), and it cannot start until the
DeepSeek-V4-Flash deployment is stopped (TASK P1.005) and the BF16 base model is resident
(TASK P1.002).
