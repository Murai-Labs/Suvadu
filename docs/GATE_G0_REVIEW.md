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
| Spec comprehension check | `notes/spec-comprehension-check.md` (local; §1–§11) | Present |
| Importable package + tests | `src/suvadu/`, `pyproject.toml`, `tests/` | Present |
| Provenance writer | `src/suvadu/provenance.py` + 14 tests | Present |
| Config contract | `src/suvadu/config.py` + 19 tests, `configs/phase0/locked-versions.yaml` | Present |
| Schema-consumer audit | `notes/schema-audits/suvadu-config.md` (local) | Present |
| Git initialized, first commit pushed | commit `cedcf86`, `github.com/Murai-Labs/Suvadu` (public) | Present |
| Ignore rules verified before push | `git check-ignore` on trackers + `notes/`; 31 files tracked, 31 on remote | Present, verified |

### Verification actually run (2026-08-16)

| Check | Command | Result |
|---|---|---|
| Source compiles | `python -m compileall -q src` | OK |
| Full suite | `python -m pytest` | **39 passed** |
| `test_config.py` | `pytest --collect-only` | 19 collected, 19 passed |
| `test_provenance.py` | `pytest --collect-only` | 14 collected, 14 passed |
| `test_import.py` | `pytest --collect-only` | 6 collected, 6 passed |
| Contract files identical | SHA256 of `CLAUDE.md` vs `AGENTS.md` | identical |
| No private files tracked | `git check-ignore` on `STATUS.md`, `CHECKPOINT.md`, `notes/**` | all ignored |
| No data-shaped files tracked | `git ls-files \| grep -E '\.(jsonl\|parquet\|safetensors\|bin\|gguf\|ckpt\|pt)$'` | none |

One disclosure to note: `tasks/atomic-task-list.md` references the local corpus path
`C:/Github/ai-traces-dataset/`. It is a path on a personal machine, not a secret, and the task
list needs it — recorded here so the disclosure is deliberate rather than accidental.

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

All Phase 0 tasks (P0.001–P0.006) are complete with the evidence tabulated above.

**Awaiting human approval only.** Approval is Ramchand's to give and is not self-granted; until
DEC-0006 is recorded, Phase 1 does not begin.

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
