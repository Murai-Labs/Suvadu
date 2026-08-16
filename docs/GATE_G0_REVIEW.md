# Gate G0 Review — Repository Skeleton

## Purpose

G0 establishes the operating system around the science: the contract, the trackers, the audit
trail, the gate chain, and an atomic task list a cold agent can resume from. It blocks all
substantive work, because a project that starts experiments before its provenance infrastructure
exists produces results it cannot later defend.

## Gate Status

Status: **Not yet approved.** Approval is recorded in `docs/DECISION_LOG.md`; this file preserves
the review evidence.

## Required Evidence

| Requirement | Evidence (path) | Status |
|-------------|-----------------|--------|
| Operating contract exists | `CLAUDE.md` | Present |
| `AGENTS.md` byte-identical to `CLAUDE.md` | SHA256 `6C71862022919E0F7955D2F993DD120980E0963D827EE3EBDF9EC53F743DE30C` on both | Present, verified |
| Codex pointer, not a third rules file | `CODEX.md` | Present |
| Public overview | `README.md` | Present |
| Trackers | `STATUS.md`, `CHECKPOINT.md`, `TASKS.md` | Present |
| Atomic task list | `tasks/atomic-task-list.md` | Present |
| Decision log seeded | `docs/DECISION_LOG.md` (DEC-0001…0005) | Present |
| Experiment log seeded | `docs/EXPERIMENT_LOG.md` | Present |
| Reproducibility requirements | `docs/REPRODUCIBILITY.md` | Present |
| Risks register | `docs/RISKS_AND_OPEN_QUESTIONS.md` (Q001–Q009) | Present |
| Runbook | `docs/RUNBOOK.md` | Present |
| Gate review files, one per gate | `docs/GATE_G0…G7_REVIEW.md` | Present |
| Append-only audit trail | `notes/` (6 files + 3 subdirs) | Present |
| Data-exclusion posture | `.gitignore` excludes `data/`, `*.jsonl`, weights, `notes/`, trackers | Present |
| Spec comprehension check | `notes/spec-comprehension-check.md` | **Missing — TASK P0.001** |
| Importable package + tests | `src/suvadu/`, `pyproject.toml`, `tests/` | **Missing — TASK P0.002** |
| Provenance writer | `src/suvadu/provenance.py` | **Missing — TASK P0.003** |
| Config contract | `src/suvadu/config.py`, `configs/phase0/locked-versions.yaml` | **Missing — TASK P0.004** |
| Git initialized, first commit | `.git/`, remote `Murai-Labs/Suvadu` | **Missing — TASK P0.006** |

## Explicit Non-Results

State plainly what has *not* happened, so no reader infers progress from file existence:

- **No training has been run.** Zero optimizer steps have been taken.
- **No data has been built.** The corpus does not exist; the July 2026 trace export is stale by
  five weeks and has not been refreshed.
- **No model has been downloaded in a trainable format.** Every Qwen3.8-27B checkpoint on any
  machine here is an inference quantization.
- **No evaluation has been run**, including of the base model. There is no reference row.
- **No measurement of any kind exists in this project.** Every number currently in the repo is
  either a computed estimate (explicitly labelled as such) or a fact read from another repo.
- The 162 GB full-FT memory figure is arithmetic from parameter counts, **not** a measured
  profile, and must not be cited as one (Q006).
- Toolchain viability on aarch64 + sm_121 is **assumed, not verified** (Q002).

## Required Next Work Before Approval

- TASK P0.001 — write `notes/spec-comprehension-check.md`.
- TASK P0.002 — package skeleton + import test.
- TASK P0.003 — provenance writer + test.
- TASK P0.004 — config contract + schema-consumer audit.
- TASK P0.006 — `git init`, verify ignore rules, first commit, create the public GitHub repo.

## Approval Template (paste into docs/DECISION_LOG.md when granted)

```
## DEC-XXXX — G0 approved
Date: <YYYY-MM-DD>
Task/Gate: G0
Decision: G0 (Repository Skeleton) approved; Phase 1 training-stack bring-up unblocked.
Rationale: <evidence summary>.
Evidence / Source Docs: docs/GATE_G0_REVIEW.md, tasks/atomic-task-list.md.
Human Approval: <name> on <date>.
```
