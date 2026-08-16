# Suvadu — Operating Contract

This file is identical at `AGENTS.md` and `CLAUDE.md`. Both Codex and Claude Code read it.
It inherits from the Murai Labs global CLAUDE.md and overrides where Suvadu-specific rules
conflict.

**This is a contract. Not guidelines, not best practices. A contract.**

## 0. Required Entry Points

Before any substantive work, read, in this order:
- `STATUS.md` — current project state.
- `CHECKPOINT.md` — exact resume point.
- `tasks/atomic-task-list.md` — canonical tasks with IDs, dependencies, acceptance criteria.
- `docs/RUNBOOK.md` — operational procedures.
- `docs/REPRODUCIBILITY.md` — run evidence requirements.
- `docs/DECISION_LOG.md` — material decisions and their rationale.
- `docs/RISKS_AND_OPEN_QUESTIONS.md` — open blockers.
- `notes/session-log.md`, `notes/stuck-log.md`, `notes/uncertainties.md` — per-session audit trail.

## 1. Identity and Stakes

- **Thesis:** Full-parameter supervised fine-tuning of Qwen3.8-27B on personal agent-session
  traces plus curated agentic/reasoning datasets yields a model measurably better than base
  Qwen3.8-27B at tool calling and coding.
- **Empirical claims:**
  1. The fine-tuned model beats base Qwen3.8-27B on tool calling (BFCL) and coding (EvalPlus)
     under a paired McNemar test.
  2. That gain is attributable to the personal traces, not merely to having run SFT on
     *something* — i.e. it survives a token-matched public-data-only control.
- **Null hypothesis:** The traces-trained model is hypothesized to outperform baselines B1–B4
  on metric M. If any of B1–B4 match it within ε = 3.0 percentage points on M, the hypothesis
  is falsified and the project concludes without publication.
- **Deliverables:** a published Hugging Face model (`Murai-Labs/Suvadu-Qwen3.8-27B`), its model
  card with honest claim boundaries, this repo's reproducible training + eval code, and a
  technical report.
- **Compute budget / timeline:** 2× NVIDIA GB10 (DGX Spark), 121 GB unified memory each, linked
  by 200G ConnectX-7 (measured 111 Gb/s RDMA, 13.6 GB/s NCCL all-reduce). Full-parameter FSDP
  across both nodes. The cluster must be exclusively available; it is shared with a
  DeepSeek-V4-Flash vLLM deployment that has to be stopped for training.

## 2. The Integrity Contract

These rules exist because silent failures waste weeks of compute. Each is non-negotiable.

### 2.1 Code claims must match code behavior
Never describe code as doing something it does not do. Verify by reading/running, not by memory.

### 2.2 Schema changes require a Schema-Consumer Audit
When you add, remove, or modify a field in any schema, dataclass, dict, or config, immediately
audit **every** consumer and write the checklist to `notes/schema-audits/<schema>.md`:
- Each new field has ≥1 consumer that reads it.
- Each removed field has no remaining references.
- Each modified field's semantics match every consumer.
"I checked" is insufficient — enumerate every consumer and the result.

### 2.3 Placeholders are forbidden in any code path affecting results
No `return 0.0  # TODO`. Raise `NotImplementedError("SUVADU-PLACEHOLDER: <what>")`.
Before any run: `grep -r "SUVADU-PLACEHOLDER" src tests` must be empty for reachable code.

### 2.4 Configurations are versioned, locked, and traceable
Every run records, before it starts: config hash, code SHA, data hash, seed, environment
(GPU, CUDA, library versions). The recording is a precondition for the run, not a TODO. If the
recording infrastructure is not in place, do not run experiments.

### 2.5 Negative results are first-class
Failed, refuted, and inconclusive results are recorded in `docs/EXPERIMENT_LOG.md` and, when they
refute a prediction, in `notes/negative-results/`. Never overwrite or delete them.

### 2.6 Reproducibility is checked, not assumed
A metric is citable only with: run ID, metric file, config hash, code SHA, data hash, seed,
environment, known limitations. Results lacking these go to `notes/untrusted-results.md` and are
excluded from the report and the model card.

### 2.7 Publication claims are held to a higher bar than internal claims
Any number that reaches the Hugging Face model card, the README, or a public post must cite a
run ID and must have been re-read from its artifact at the moment of writing — not recalled from
an earlier session. A comparative claim ("better than base") additionally requires the paired
test, its p-value, and its stated sensitivity floor.

## 3. Cheap-Baseline-Falsification Gate (G4 — MANDATORY)

**No phase past G4 begins until the simplest non-mechanism explanation has been run on metric M
and failed to explain the effect.** Argument that a baseline "isn't fair" is not sufficient —
the baseline must be executed and must fail.

Metric **M** = BFCL overall accuracy (tool calling) and EvalPlus HumanEval+/MBPP+ pass@1
(coding), evaluated as a paired comparison with McNemar's exact test.

Mandatory baselines for this project:
- **B1 — token-matched public-mixture-only SFT.** Identical recipe, identical token budget,
  public datasets only, zero personal traces. If B1 matches the traces model, the traces added
  nothing beyond data you would have used anyway. *This is the primary null.*
- **B2 — prompt-only / RAG, zero training.** Base Qwen3.8-27B with the global CLAUDE.md and
  retrieved trace excerpts in context. If B2 matches, no training was needed at all. This
  baseline was identified in July 2026 and never run; it is now a task, not an instinct.
- **B3 — traces at 10% subsample.** If B3 matches the full-traces run, corpus scale is not the
  active ingredient and the effect is cheaper than claimed.
- **B4 — LoRA instead of full-parameter FT.** If B4 matches, full-parameter training did not
  earn its cluster cost and the recipe should be downgraded.

**ε = 3.0 percentage points at n = 200.** Justification: the paired McNemar harness reused from
`Murai-Labs/qwen3.8-27b-rtx5090` has a *measured* sensitivity floor of 6 discordant items
(documented in that repo's `bench/LEADERBOARD.md`), independent of n. At n = 200 that is exactly
3.0 pp, so a smaller difference is invisible to the test. Resolving finer requires raising n;
computing the required n is TASK P3.004, not an assumption.

If any cheap baseline matches or exceeds the traces model within ε on M, **G5+ is blocked; the
project re-scopes or terminates.**

## 4. Long-Running Session Discipline

- Checkpoint progress to `STATUS.md` + `CHECKPOINT.md` before every state-changing commit.
- When stuck, append to `notes/stuck-log.md` (task, attempts, failures, hypothesis) and escalate.
- Claims of completion require evidence (a run ID + metrics path), not file existence.
- Any run >30s emits progress every ≤100 steps: step/total, elapsed, ETA (better: loss,
  throughput, memory). A JSON file written each epoch is not progress.
- Runs >30min write a resume-state file each epoch and validate config/seed on restart.

## 5. Subagent Loop Discipline

- Log every subagent invocation to `notes/subagent-log.md` (timestamp, task, prompt summary,
  output summary, action taken).
- Subagents share the parent's blindspots — do not treat agreement among them as verification.
- A subagent that cannot complete its task escalates; it does not fabricate a plausible result.

## 6. Forbidden Patterns

- Hardcoded numeric dtypes scattered across code (centralize in a config/dtype policy).
  Use `dtype=...`, never the deprecated `torch_dtype=...`.
- Evaluation data leaking into training pipelines. **This project is at elevated risk:** the
  trace corpus contains real sessions that may quote HumanEval/MBPP/BFCL problems verbatim.
  Decontamination is a gate (G2), not a courtesy.
- Silent fallbacks that mask failure (catch-and-continue without logging).
- Result files without the 5 provenance identifiers.
- Committing any trace-derived data, merged corpus, or model weight into this repo. The repo is
  public.

## 7. Decision Gates

Gate approval is recorded in `docs/DECISION_LOG.md`, with evidence preserved in the matching
`docs/GATE_Gn_REVIEW.md`.

- **G0** — repo skeleton, governance, trackers, task list exist.
- **G1** — training stack verified: BF16 base resident on the cluster, 2-node FSDP completes ≥1
  optimizer step, loss decreases over a short smoke run, throughput and peak memory measured.
- **G2** — data frozen: traces re-exported to current date, mixture merged, **decontaminated
  against BFCL/EvalPlus**, licences reviewed, corpus hash recorded.
- **G3** — evaluation harness frozen and **base Qwen3.8-27B measured** on M. No training result
  is interpretable without this reference row.
- **G4** — **cheap-baseline falsification complete** (Section 3). Hard stop.
- **G5** — launch approval for the main full-parameter run (the single most expensive job).
- **G6** — capability gate: fine-tuned model beats base on M under paired McNemar, with p and
  sensitivity floor stated.
- **G7** — memorization audit clean → publication approval. See Section 11.

## 8. File Conventions

- `notes/` files are append-only. Past entries are never edited or deleted; a wrong entry is
  corrected by appending a new one that references it.
- Repo is **text-free**: raw/large/copyrighted data and checkpoints are gitignored; commit
  manifests, configs, code, docs, reports only.
- Run IDs are monotonic/descriptive and never reused, even for failed attempts.
- **This repo is public.** Per the Murai Labs global rule, internal tracking files
  (`STATUS.md`, `CHECKPOINT.md`, `GAPS.md`, and everything under `notes/`) are gitignored and
  stay local. `docs/` *is* committed — the decision log and gate reviews are the public research
  record and are what make the published claim auditable.

## 9. Verification

After edits, run the most relevant check available; never report completion on file existence
alone:
- Markdown-only: re-read changed files; check links/headings.
- Code: `python -m compileall src` and focused `pytest -k <area>`.
- Training/data scripts: a short smoke run before any long job.

## 10. Conflict Resolution

Precedence: Section 11 (publication safety) > Section 2 (integrity) > Section 3 (cheap-baseline
gate) > Section 7 (gates) > Sections 4–6 > Section 8. When rules conflict, the higher-precedence
rule wins and the conflict is noted in `docs/DECISION_LOG.md`.

## 11. Publication Safety (G7) — highest precedence

The training corpus is derived from real agent sessions on a personal machine. Those sessions
contain private source code pasted into tool outputs, which the July 2026 scrub explicitly did
**not** remove. Full-parameter fine-tuning memorizes; publishing the weights publishes whatever
was memorized.

Binding rules:

1. **No weights are pushed to Hugging Face before G7 passes.** Not as private, not as a draft,
   not "temporarily".
2. The G7 audit must include, at minimum: targeted extraction probes for known-private strings
   from Uyir, TamilLM, Pinnal-Core, ChittiOS, Kickoff-bot (Certinia/Salesforce, client-adjacent),
   HouseBuild and career-ops; a secret-pattern sweep of generated output; and a
   training-data-extraction attack at non-zero temperature with prefix prompting.
3. A **PASS requires zero verbatim reproduction** of private source beyond a pre-registered
   n-gram threshold, recorded in `docs/GATE_G7_REVIEW.md` before the audit runs.
4. If the audit fails, the remedy is corpus repair and retraining — **not** a disclaimer in the
   model card.
5. Third-party data licensing is part of G7. Several intended source datasets display no licence
   and are distillations of other vendors' model outputs; their redistribution status must be
   resolved in writing before publication.
