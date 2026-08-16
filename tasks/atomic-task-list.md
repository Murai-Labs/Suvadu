# Suvadu — Atomic Task List

Canonical task system. Update this file plus `STATUS.md`, `CHECKPOINT.md`, and affected docs on
any status change.

**Spec:** this project has no external spec document. The spec is the design conversation of
2026-08-16, captured in `notes/spec-comprehension-check.md` and `docs/DECISION_LOG.md` DEC-0001.
Task `Why` fields cite `spec §` numbers that refer to sections of the comprehension check.

**Gate chain:** G0 skeleton → G1 stack → G2 data → G3 eval+base → **G4 cheap baselines** →
G5 launch → G6 capability → G7 memorization/publication.

---

## Phase 0 — Skeleton & Comprehension (Gate G0)

#### TASK P0.001: Write the spec comprehension check
- **What:** A written statement of thesis, claims, null hypothesis, gates and baselines that any
  future agent can resume from cold.
- **Where:** `notes/spec-comprehension-check.md`
- **Why:** spec §all · AGENTS.md §1 · there is no external spec doc, so this file *is* the spec.
- **Inputs:** none
- **Acceptance criteria:**
  1. Thesis, both empirical claims, and the null hypothesis are stated in ≤15 lines.
  2. Gates G0–G7 are each listed with the evidence they require.
  3. Baselines B1–B4 are enumerated with what each would prove if it won.
  4. ε = 3.0 pp is stated with its derivation from the 6-discordant-item floor.
- **Evidence of completion:** `notes/spec-comprehension-check.md`
- **Validation:** human read; cross-check against `CLAUDE.md` §1 and §3.
- **Measurements / logs:** n/a
- **Dependencies:** none
- **Blocking gate:** G0
- **Estimated effort:** 1
- **Done:** [ ]

#### TASK P0.002: Establish package skeleton and tests
- **What:** Installable `src/suvadu/` package with version, plus an import/layout smoke test.
- **Where:** `src/suvadu/__init__.py`, `pyproject.toml`, `tests/test_import.py`
- **Why:** spec §6 · AGENTS.md §9 · everything downstream depends on an importable package.
- **Inputs:** none
- **Acceptance criteria:**
  1. `python -c "import suvadu; print(suvadu.__version__)"` succeeds.
  2. `tests/test_import.py` asserts the expected top-level layout and passes.
- **Evidence of completion:** passing `pytest -q`.
- **Validation:** `python -m compileall src && pytest -q`
- **Measurements / logs:** n/a
- **Dependencies:** none
- **Blocking gate:** G0
- **Estimated effort:** 2
- **Done:** [ ]

#### TASK P0.003: Implement run-provenance writer
- **What:** A writer recording the 5 identifiers (config hash, code SHA, data hash, seed,
  environment) into every run directory.
- **Where:** `src/suvadu/provenance.py`, `tests/test_provenance.py`
- **Why:** spec §7 · AGENTS.md §2.4, §2.6 · no run is permitted without provenance.
- **Inputs:** P0.002
- **Acceptance criteria:**
  1. Writing creates exactly one `manifest.json` per run dir containing all 5 identifier
     categories, with the environment record capturing GPU, CUDA, Python and package versions.
  2. A validator rejects a run dir missing any identifier, with a typed error naming which.
  3. Environment capture works on aarch64 (GB10) — not only on x86.
- **Evidence of completion:** `src/suvadu/provenance.py`, passing test.
- **Validation:** `pytest -k provenance`
- **Measurements / logs:** n/a
- **Dependencies:** P0.002
- **Blocking gate:** G0
- **Estimated effort:** 3
- **Done:** [ ]

#### TASK P0.004: Lock the config contract
- **What:** A versioned config schema plus a locked dependency-version file.
- **Where:** `src/suvadu/config.py`, `configs/phase0/locked-versions.yaml`, `tests/test_config.py`
- **Why:** spec §7 · AGENTS.md §2.4 · rejects silent config drift before expensive runs.
- **Inputs:** P0.002
- **Acceptance criteria:**
  1. Loading an out-of-contract config raises a typed validation error naming the bad field.
  2. A schema-consumer audit is written enumerating every consumer of the config schema.
  3. `dtype` is expressed as `dtype=` throughout; `grep -rn torch_dtype src` returns nothing.
- **Evidence of completion:** `notes/schema-audits/suvadu-config.md`, passing test.
- **Validation:** `pytest -k config && grep -rn torch_dtype src ; test $? -ne 0`
- **Measurements / logs:** n/a
- **Dependencies:** P0.002
- **Blocking gate:** G0
- **Estimated effort:** 3
- **Done:** [ ]

#### TASK P0.005: Author the G0 gate review and close G0
- **What:** The G0 review document recording what exists and, explicitly, what does not.
- **Where:** `docs/GATE_G0_REVIEW.md`, `docs/DECISION_LOG.md`
- **Why:** spec §8 · AGENTS.md §7 · gates are evidence checkpoints, not announcements.
- **Inputs:** P0.001, P0.002, P0.003, P0.004
- **Acceptance criteria:**
  1. Every required-evidence row is marked Present with a real path, or Missing.
  2. The "Explicit Non-Results" section states that zero runs and zero measurements exist.
  3. Human approval recorded in `docs/DECISION_LOG.md`.
- **Evidence of completion:** `docs/GATE_G0_REVIEW.md` + DECISION_LOG entry.
- **Validation:** human review.
- **Measurements / logs:** n/a
- **Dependencies:** P0.001, P0.002, P0.003, P0.004
- **Blocking gate:** G0
- **Estimated effort:** 1
- **Done:** [ ]

#### TASK P0.006: Initialize git and create the public GitHub repo
- **What:** Local git repo plus `Murai-Labs/Suvadu` on GitHub, public, first commit pushed.
- **Where:** `.git/`, remote `origin`
- **Why:** spec §9 · AGENTS.md §8 · the public research record starts here.
- **Inputs:** P0.005
- **Acceptance criteria:**
  1. `git status --porcelain --ignored` confirms `data/`, `notes/`, `STATUS.md`,
     `CHECKPOINT.md` are ignored, not tracked.
  2. `git ls-files | grep -E '\.(jsonl|safetensors|bin|gguf)$'` returns nothing.
  3. First commit message is `chore: scaffold Suvadu research repo (G0 skeleton)`.
- **Evidence of completion:** commit SHA, remote URL.
- **Validation:** `git ls-files | wc -l` reviewed by a human before push.
- **Measurements / logs:** n/a
- **Dependencies:** P0.005
- **Blocking gate:** G0
- **Estimated effort:** 1
- **Done:** [ ]

---

## Phase 1 — Training Stack Bring-Up (Gate G1)

> Purpose: prove a 27B full-parameter step is physically possible on this cluster **before**
> spending days on data. If the toolchain does not exist for aarch64 + sm_121, the whole project
> re-scopes to LoRA and we want to know that now, not after the corpus is built.

#### TASK P1.001: Verify the training toolchain on aarch64/sm_121
- **What:** A written environment report proving (or disproving) that transformers v5 + the
  chosen trainer import and see the GPU on a GB10 node.
- **Where:** `docs/decisions/0001-training-stack.md`, `configs/phase1/env-report.md`
- **Why:** spec §3 · AGENTS.md §1 · global rule: aarch64+Blackwell wheels are spotty.
- **Inputs:** P0.006
- **Acceptance criteria:**
  1. `transformers.__version__` is v5+ and `AutoConfig.from_pretrained("Qwen/Qwen3.8-27B")`
     resolves `model_type: qwen3_5` without a trust_remote_code failure.
  2. `torch.cuda.is_available()` is True and the reported capability is sm_121.
  3. The report states whether the run is bare-metal or containerized, and pins image/wheel IDs.
- **Evidence of completion:** `configs/phase1/env-report.md` with pasted command output.
- **Validation:** run on `Murailabs-Spark` via base64-over-SSH; paste literal output.
- **Measurements / logs:** versions, capability, container digest.
- **Dependencies:** P0.006
- **Blocking gate:** G1
- **Estimated effort:** 4
- **Done:** [ ]

#### TASK P1.002: Download BF16 Qwen3.8-27B to the cluster
- **What:** The trainable BF16 base checkpoint resident on `spark-1003`, hash-verified.
- **Where:** `spark-1003:~/models/Qwen3.8-27B/`, manifest at `configs/phase1/base-model.json`
- **Why:** spec §3 · AGENTS.md §2.4 · every local copy today is an inference quantization.
- **Inputs:** P1.001
- **Acceptance criteria:**
  1. `hf download Qwen/Qwen3.8-27B` completes; on-disk size is within 5% of the expected ~54 GB.
  2. The revision SHA is pinned and recorded in `configs/phase1/base-model.json`.
  3. A load test instantiates the config and tokenizer (not full weights) without error.
- **Evidence of completion:** `configs/phase1/base-model.json` with revision SHA and byte count.
- **Validation:** `du -sb`, `hf` revision output, tokenizer round-trip test.
- **Measurements / logs:** download duration, final size, revision SHA.
- **Dependencies:** P1.001
- **Blocking gate:** G1
- **Estimated effort:** 3
- **Done:** [ ]

#### TASK P1.003: Measure the real memory profile of a 27B optimizer step
- **What:** A measured (not computed) memory profile for full-parameter FSDP on 2 nodes.
- **Where:** `src/suvadu/cli/memprobe.py`, `runs/phase1-memprobe-001/`
- **Why:** spec §3 · AGENTS.md §2.6 · the 162 GB figure in the README is arithmetic, not a
  measurement, and must not be cited as one.
- **Inputs:** P1.002, **cluster free** (DeepSeek stopped — see P1.005)
- **Acceptance criteria:**
  1. Peak per-node memory is recorded for at least: full FT + 8-bit optimizer, and LoRA.
  2. The run either completes one optimizer step or fails with a recorded OOM trace.
  3. Result contradicts or confirms the computed estimate, and the README is corrected if wrong.
- **Evidence of completion:** run-id `phase1-memprobe-001`, `metrics.json`.
- **Validation:** smoke run with `--max-steps 1`.
- **Measurements / logs:** peak RSS/HBM per node, step wall-clock, NCCL bandwidth achieved.
- **Dependencies:** P1.002, P1.005
- **Blocking gate:** G1
- **Estimated effort:** 6
- **Done:** [ ]

#### TASK P1.004: Implement the training entrypoint with progress reporting
- **What:** A single training CLI that runs FSDP across both nodes and emits progress.
- **Where:** `src/suvadu/cli/train.py`, `configs/phase1/smoke.yaml`
- **Why:** spec §4 · AGENTS.md §4 · silent jobs are indistinguishable from hangs.
- **Inputs:** P0.003, P0.004, P1.001
- **Acceptance criteria:**
  1. Emits `step/total`, elapsed, ETA, loss, throughput and peak memory at least every 100 steps.
  2. Writes a provenance manifest *before* the first step, and refuses to start without one.
  3. Writes resume state each epoch and validates config hash + seed on restart.
  4. `grep -r "SUVADU-PLACEHOLDER" src tests` is empty for reachable code.
- **Evidence of completion:** `src/suvadu/cli/train.py`, passing `pytest -k train`.
- **Validation:** `python -m compileall src && pytest -k train`
- **Measurements / logs:** n/a (harness task)
- **Dependencies:** P0.003, P0.004, P1.001
- **Blocking gate:** G1
- **Estimated effort:** 8
- **Done:** [ ]

#### TASK P1.005: Stop the DeepSeek-V4-Flash deployment and record how to restore it
- **What:** A written restore procedure, executed shutdown, and confirmation the cluster is free.
- **Where:** `docs/RUNBOOK.md` (restore section), `notes/session-log.md`
- **Why:** spec §3 · AGENTS.md §4 · the cluster is shared; taking it down must be reversible.
- **Inputs:** P1.002, P1.004 (do not stop the endpoint before both exist)
- **Acceptance criteria:**
  1. The exact relaunch command for DeepSeek-V4-Flash TP=2 is captured **before** shutdown,
     including the `--headless` flag required on worker rank 1.
  2. Post-shutdown `free -g` shows ≥110 GB available on both nodes.
  3. Shutdown is human-approved on the day it happens (approval already granted 2026-08-16).
- **Evidence of completion:** RUNBOOK restore section + pasted `free -g` output.
- **Validation:** `ssh -T -n Murailabs-Spark "free -g"` on both nodes.
- **Measurements / logs:** available memory per node, timestamp.
- **Dependencies:** P1.002, P1.004
- **Blocking gate:** G1
- **Estimated effort:** 2
- **Done:** [ ]

#### TASK P1.006: G1 decision — training stack verified
- **What:** The G1 gate review recording that a real optimizer step ran and loss decreased.
- **Where:** `docs/GATE_G1_REVIEW.md`, `docs/DECISION_LOG.md`
- **Why:** spec §8 · AGENTS.md §7
- **Inputs:** P1.003, P1.004
- **Acceptance criteria:**
  1. A smoke run of ≥50 steps shows monotone-ish decreasing loss, with the loss curve archived.
  2. Measured throughput (tokens/s) and peak memory are recorded, and the projected wall-clock
     for the main run is computed from them.
  3. If full FT proved infeasible, the gate records the re-scope to LoRA explicitly rather than
     silently changing the plan.
- **Evidence of completion:** `docs/GATE_G1_REVIEW.md` + DECISION_LOG entry.
- **Validation:** human review.
- **Measurements / logs:** loss curve, tok/s, peak memory, projected main-run hours.
- **Dependencies:** P1.003, P1.004, P1.005
- **Blocking gate:** G1
- **Estimated effort:** 2
- **Done:** [ ]

---

## Phase 2 — Data Build & Freeze (Gate G2)

#### TASK P2.001: Re-export agent traces to current date
- **What:** A refreshed raw trace export covering sessions through the export date.
- **Where:** `C:/Github/ai-traces-dataset/` (external, gitignored), manifest at
  `configs/phase2/traces-export.json`
- **Why:** spec §2 · AGENTS.md §2.4 · the existing export is dated 2026-07-13 and misses ~5
  weeks of sessions.
- **Inputs:** P0.006
- **Acceptance criteria:**
  1. Export runs `build_traces_dataset.py` and reports conversation and message counts.
  2. New counts exceed the 2026-07-13 baseline (1,106 convos / ~202k messages) and the delta is
     stated.
  3. `scrub_report.json` is regenerated and its secret-catch counts recorded.
- **Evidence of completion:** `configs/phase2/traces-export.json` with counts + corpus hash.
- **Validation:** re-read the manifest at reporting time; spot-check 3 conversations by hand.
- **Measurements / logs:** convo count, message count, token count, scrub counts.
- **Dependencies:** P0.006
- **Blocking gate:** G2
- **Estimated effort:** 3
- **Done:** [ ]

#### TASK P2.002: Audit trace scrubbing for private source code
- **What:** A hand audit quantifying how much private source code survives the scrub.
- **Where:** `notes/integrity-gaps.md`, `docs/decisions/0002-corpus-safety.md`
- **Why:** spec §5 · AGENTS.md §11 · the July scrub explicitly did not remove pasted source.
- **Inputs:** P2.001
- **Acceptance criteria:**
  1. A random sample of ≥50 conversations is read by hand and classified for private-code content.
  2. The rate is reported with its denominator, and ≥3 instances are quoted (locally, not in the
     public repo) as evidence.
  3. A decision is recorded: strip tool outputs, strip by repo allowlist, or accept the risk.
- **Evidence of completion:** `docs/decisions/0002-corpus-safety.md`.
- **Validation:** human read of the sample.
- **Measurements / logs:** contaminated-sample rate, per-repo breakdown.
- **Dependencies:** P2.001
- **Blocking gate:** G2, G7
- **Estimated effort:** 6
- **Done:** [ ]

#### TASK P2.003: Resolve licensing for third-party datasets
- **What:** A written licence determination for every external dataset in the mixture.
- **Where:** `docs/decisions/0003-dataset-licences.md`
- **Why:** spec §2 · AGENTS.md §11.5 · several intended sources display no licence.
- **Inputs:** P0.006
- **Acceptance criteria:**
  1. Each dataset is listed with repo id, row count, stated licence, and redistribution status.
  2. Datasets with no stated licence are marked EXCLUDED or escalated for a human decision —
     never silently included.
  3. Vendor-output-distillation status is recorded per dataset (ToS exposure).
- **Evidence of completion:** `docs/decisions/0003-dataset-licences.md`.
- **Validation:** human review; each claim cites the dataset page fetched on a stated date.
- **Measurements / logs:** n/a
- **Dependencies:** P0.006
- **Blocking gate:** G2, G7
- **Estimated effort:** 3
- **Done:** [ ]

#### TASK P2.004: Build the corpus merge and formatting pipeline
- **What:** One CLI that merges traces + approved datasets into a single chat-formatted corpus.
- **Where:** `src/suvadu/data/build_corpus.py`, `configs/phase2/mixture.yaml`
- **Why:** spec §2 · AGENTS.md §2.3
- **Inputs:** P2.001, P2.003
- **Acceptance criteria:**
  1. Output conforms to the Qwen3.8 chat template, verified by rendering 5 samples and diffing
     against the tokenizer's own `apply_chat_template`.
  2. Mixture ratios are declared in `mixture.yaml`, not hardcoded.
  3. The corpus hash is deterministic given the same inputs and seed.
- **Evidence of completion:** corpus manifest with hash + per-source token counts.
- **Validation:** `pytest -k build_corpus`
- **Measurements / logs:** tokens per source, total tokens, sequence-length histogram.
- **Dependencies:** P2.001, P2.003
- **Blocking gate:** G2
- **Estimated effort:** 8
- **Done:** [ ]

#### TASK P2.005: Handle the `<think>` / reasoning-trace format collision
- **What:** A decision plus implementation for reasoning content in the training corpus.
- **Where:** `docs/decisions/0004-thinking-tokens.md`, `src/suvadu/data/build_corpus.py`
- **Why:** spec §2 · the qwen3.8-27b-rtx5090 findings · Qwen3.8's template handles
  `reasoning_content` specially and `preserve_thinking` alters rendering; the r0b0tlab and
  11-47 datasets ship `<think>` blocks inline in the response field.
- **Inputs:** P2.004
- **Acceptance criteria:**
  1. A decision is recorded on whether thinking is trained on, stripped, or moved to
     `reasoning_content`.
  2. Rendered training samples are byte-compared against inference-time rendering; any mismatch
     between train and serve formatting is documented as a known train/serve skew.
- **Evidence of completion:** `docs/decisions/0004-thinking-tokens.md` + passing format test.
- **Validation:** `pytest -k chat_template`
- **Measurements / logs:** n/a
- **Dependencies:** P2.004
- **Blocking gate:** G2
- **Estimated effort:** 4
- **Done:** [ ]

#### TASK P2.006: Decontaminate the corpus against the evaluation sets
- **What:** A decontamination pass removing BFCL/EvalPlus overlap from training data.
- **Where:** `src/suvadu/data/decontaminate.py`, `configs/phase2/decontam-report.json`
- **Why:** spec §5 · AGENTS.md §6 · **elevated risk**: real sessions plausibly quote HumanEval
  and MBPP problems verbatim, which would invalidate the entire capability claim.
- **Inputs:** P2.004, P3.001
- **Acceptance criteria:**
  1. n-gram overlap detection runs against every eval set used at G3/G6, with n pre-registered.
  2. The number of removed sequences is reported per eval set, with ≥5 removed examples read by
     hand to confirm they are genuine contamination and not false positives.
  3. Post-decontamination overlap is zero at the chosen threshold.
- **Evidence of completion:** `configs/phase2/decontam-report.json`.
- **Validation:** `pytest -k decontaminate`; hand-read of sampled removals.
- **Measurements / logs:** overlap counts per eval set, before and after.
- **Dependencies:** P2.004, P3.001
- **Blocking gate:** G2, G6
- **Estimated effort:** 6
- **Done:** [ ]

#### TASK P2.007: Build the token-matched control corpus (B1)
- **What:** The public-data-only corpus, token-matched to the traces corpus.
- **Where:** `configs/phase2/mixture-control.yaml`, control corpus manifest
- **Why:** spec §3 · AGENTS.md §3 · B1 is the primary null and must be built *with* the treatment
  corpus so the two are matched by construction, not retrofitted.
- **Inputs:** P2.004, P2.006
- **Acceptance criteria:**
  1. Control corpus total token count is within 2% of the traces corpus.
  2. It contains zero personal-trace sequences, verified by source-tag audit.
  3. Sequence-length distributions of the two corpora are compared and the difference reported.
- **Evidence of completion:** control corpus manifest with hash + token count.
- **Validation:** `pytest -k control_corpus`
- **Measurements / logs:** token counts both corpora, length histograms.
- **Dependencies:** P2.004, P2.006
- **Blocking gate:** G2, G4
- **Estimated effort:** 4
- **Done:** [ ]

#### TASK P2.008: G2 decision — freeze the data
- **What:** The G2 gate review freezing corpus hashes for all arms.
- **Where:** `docs/GATE_G2_REVIEW.md`, `docs/DECISION_LOG.md`
- **Why:** spec §8 · AGENTS.md §7
- **Inputs:** P2.002, P2.003, P2.005, P2.006, P2.007
- **Acceptance criteria:**
  1. Every arm's corpus hash is recorded and immutable from this point.
  2. Decontamination report shows zero residual overlap.
  3. Licence determination shows zero unlicensed datasets included.
- **Evidence of completion:** `docs/GATE_G2_REVIEW.md` + DECISION_LOG entry.
- **Validation:** human review.
- **Measurements / logs:** corpus hashes, token counts per arm.
- **Dependencies:** P2.002, P2.003, P2.005, P2.006, P2.007
- **Blocking gate:** G2
- **Estimated effort:** 2
- **Done:** [ ]

---

## Phase 3 — Evaluation Harness & Base Measurement (Gate G3)

#### TASK P3.001: Select and pin the evaluation suites
- **What:** A written, versioned choice of the exact eval sets and their revisions.
- **Where:** `docs/decisions/0005-evaluation-suites.md`, `configs/phase3/evals.yaml`
- **Why:** spec §3 · AGENTS.md §2.6 · the metric must be fixed before any arm is trained.
- **Inputs:** P0.006
- **Acceptance criteria:**
  1. Tool calling and coding suites are named with dataset revision SHAs and split definitions.
  2. Scoring is defined exactly (exact match, pass@1, harness version).
  3. The suites are the same ones the eventual model card and report will cite.
- **Evidence of completion:** `configs/phase3/evals.yaml`.
- **Validation:** human review.
- **Measurements / logs:** n/a
- **Dependencies:** P0.006
- **Blocking gate:** G3
- **Estimated effort:** 3
- **Done:** [ ]

#### TASK P3.002: Port the paired-McNemar harness from qwen3.8-27b-rtx5090
- **What:** The paired comparison harness, reused rather than rewritten.
- **Where:** `src/suvadu/eval/compare.py`, `tests/test_compare.py`
- **Why:** spec §3 · AGENTS.md §2.6 · that harness already encodes the discordant-item logic and
  its measured sensitivity floor; rewriting it would discard validated work.
- **Inputs:** P3.001
- **Acceptance criteria:**
  1. Reproduces a known result from the source repo's `bench/results/` to the same p-value.
  2. Reports both the McNemar p and the discordant-item counts, never accuracy alone.
  3. Refuses to compare runs whose provenance manifests are missing.
- **Evidence of completion:** passing `pytest -k compare` including the reproduction test.
- **Validation:** `pytest -k compare`
- **Measurements / logs:** n/a
- **Dependencies:** P3.001
- **Blocking gate:** G3
- **Estimated effort:** 4
- **Done:** [ ]

#### TASK P3.003: Measure base Qwen3.8-27B — the reference row
- **What:** The base model's scores on M, with full provenance.
- **Where:** `runs/phase3-base-001/`, `docs/EXPERIMENT_LOG.md`
- **Why:** spec §3 · AGENTS.md §2.6 · no training result is interpretable without this row.
- **Inputs:** P1.002, P3.001, P3.002
- **Acceptance criteria:**
  1. Base scores recorded on every suite in `configs/phase3/evals.yaml`.
  2. Per-item results are persisted (not just totals) so later comparisons can be paired.
  3. Before reporting any aggregate, ≥3 individual items are read to confirm the scorer works.
- **Evidence of completion:** run-id `phase3-base-001`, `metrics.json`, per-item results file.
- **Validation:** re-run one suite at a second seed; report the spread.
- **Measurements / logs:** per-suite accuracy, per-item outcomes, wall-clock.
- **Dependencies:** P1.002, P3.001, P3.002
- **Blocking gate:** G3
- **Estimated effort:** 6
- **Done:** [ ]

#### TASK P3.004: Compute the n required to resolve ε
- **What:** A power calculation fixing the eval sample size.
- **Where:** `notes/decision-gates/g4-cheap-baseline.md` (n section), `docs/DECISION_LOG.md`
- **Why:** spec §3 · AGENTS.md §3 · ε = 3.0 pp is the floor *at n=200*; if we want to detect
  smaller effects we must raise n, and that must be decided before results are seen.
- **Inputs:** P3.003
- **Acceptance criteria:**
  1. The required n for the target detectable difference is computed and stated.
  2. If the chosen n exceeds the suite size, that limitation is recorded as a claim boundary.
  3. Recorded with a date preceding any Phase 4 run.
- **Evidence of completion:** DECISION_LOG entry with the calculation.
- **Validation:** human review; timestamps precede Phase 4.
- **Measurements / logs:** n/a
- **Dependencies:** P3.003
- **Blocking gate:** G3, G4
- **Estimated effort:** 2
- **Done:** [ ]

#### TASK P3.005: G3 decision — freeze the evaluation
- **What:** The G3 gate review freezing metric, harness and base reference.
- **Where:** `docs/GATE_G3_REVIEW.md`, `docs/DECISION_LOG.md`
- **Why:** spec §8 · AGENTS.md §7
- **Inputs:** P3.001, P3.002, P3.003, P3.004
- **Acceptance criteria:**
  1. Metric M, harness version, suite revisions, n, and ε are all frozen and recorded together.
  2. Base reference row is present with provenance.
- **Evidence of completion:** `docs/GATE_G3_REVIEW.md` + DECISION_LOG entry.
- **Validation:** human review.
- **Measurements / logs:** the frozen base row.
- **Dependencies:** P3.001, P3.002, P3.003, P3.004
- **Blocking gate:** G3
- **Estimated effort:** 1
- **Done:** [ ]

---

## Phase 4 — Cheap-Baseline Falsification (Gate G4, MANDATORY)

> Null hypothesis: the traces-trained model is hypothesized to beat baselines B1–B4 on metric M.
> **If any baseline matches it within ε = 3.0 pp on M, G5+ is blocked; the project re-scopes or
> terminates.** These are real, executed runs — not arguments.
>
> Note on ordering: B2 is by far the cheapest (no training at all). Run it first. If prompting
> alone matches the target, the project can conclude before a single GPU-hour is spent on SFT.

#### TASK P4.001: Run baseline B2 — prompt-only / RAG, zero training
- **What:** Base Qwen3.8-27B measured on M with global CLAUDE.md + retrieved trace context.
- **Where:** `src/suvadu/cli/run_baseline_b2.py`, `configs/phase4/baseline_b2.yaml`
- **Why:** spec §3 · AGENTS.md §3 · identified as the cheap baseline in July 2026 and never run.
- **Inputs:** P3.005
- **Acceptance criteria:**
  1. Runs on the same suites, split and scorer as `phase3-base-001`.
  2. Retrieval is described exactly (index, k, chunking) and is reproducible from the config.
  3. Emits provenance and progress every ≤100 steps.
- **Evidence of completion:** run-id `phase4-baseline-b2-001`, `metrics.json`.
- **Validation:** smoke run then full run; `pytest -k baseline_b2`
- **Measurements / logs:** metric M, retrieval hit stats, seed.
- **Dependencies:** P3.005
- **Blocking gate:** G4
- **Estimated effort:** 8
- **Done:** [ ]

#### TASK P4.002: Train and measure baseline B1 — token-matched public-only SFT
- **What:** The primary null: same recipe and token budget, public data only.
- **Where:** `configs/phase4/baseline_b1.yaml`, `runs/phase4-baseline-b1-001/`
- **Why:** spec §3 · AGENTS.md §3 · isolates *your traces* from *any* SFT.
- **Inputs:** P2.007, P3.005, P1.006
- **Acceptance criteria:**
  1. Hyperparameters are byte-identical to the treatment config except the corpus path.
  2. Token count consumed is within 2% of the treatment run.
  3. Measured on M with per-item results persisted for pairing.
- **Evidence of completion:** run-id `phase4-baseline-b1-001`, `metrics.json`.
- **Validation:** config diff against treatment shows only the corpus path differs.
- **Measurements / logs:** loss curve, tokens consumed, metric M per suite.
- **Dependencies:** P2.007, P3.005, P1.006
- **Blocking gate:** G4
- **Estimated effort:** 16
- **Done:** [ ]

#### TASK P4.003: Train and measure baseline B3 — traces at 10% subsample
- **What:** The scale null.
- **Where:** `configs/phase4/baseline_b3.yaml`, `runs/phase4-baseline-b3-001/`
- **Why:** spec §3 · AGENTS.md §3
- **Inputs:** P2.008, P3.005, P1.006
- **Acceptance criteria:**
  1. Subsample is drawn with a recorded seed and is a strict subset of the treatment corpus.
  2. Measured on M with per-item results persisted.
- **Evidence of completion:** run-id `phase4-baseline-b3-001`, `metrics.json`.
- **Validation:** subset check by hash membership.
- **Measurements / logs:** metric M, tokens consumed, seed.
- **Dependencies:** P2.008, P3.005, P1.006
- **Blocking gate:** G4
- **Estimated effort:** 10
- **Done:** [ ]

#### TASK P4.004: Train and measure baseline B4 — LoRA instead of full FT
- **What:** The regime null: did full-parameter training earn its cost?
- **Where:** `configs/phase4/baseline_b4.yaml`, `runs/phase4-baseline-b4-001/`
- **Why:** spec §3 · AGENTS.md §3 · LoRA fits one node, so this is the cheapest training arm.
- **Inputs:** P2.008, P3.005, P1.006
- **Acceptance criteria:**
  1. Target modules are chosen deliberately and justified — note that only 16 of 64 layers are
     attention; the Gated DeltaNet layers must be explicitly included or excluded with a reason.
  2. Same corpus and token budget as the treatment run.
  3. Measured on M with per-item results persisted.
- **Evidence of completion:** run-id `phase4-baseline-b4-001`, `metrics.json`.
- **Validation:** adapter target-module list recorded in the config and in the gate review.
- **Measurements / logs:** metric M, trainable-parameter count, peak memory.
- **Dependencies:** P2.008, P3.005, P1.006
- **Blocking gate:** G4
- **Estimated effort:** 10
- **Done:** [ ]

#### TASK P4.005: Train the treatment arm at baseline scale
- **What:** The traces-trained model at the same scale as B1–B4, for a fair G4 comparison.
- **Where:** `configs/phase4/treatment.yaml`, `runs/phase4-treatment-001/`
- **Why:** spec §3 · AGENTS.md §3 · the gate compares like with like; the *main* run is G5.
- **Inputs:** P2.008, P3.005, P1.006
- **Acceptance criteria:**
  1. Identical hyperparameters to B1 except the corpus.
  2. Measured on M with per-item results persisted.
- **Evidence of completion:** run-id `phase4-treatment-001`, `metrics.json`.
- **Validation:** config diff vs B1.
- **Measurements / logs:** loss curve, metric M, tokens consumed.
- **Dependencies:** P2.008, P3.005, P1.006
- **Blocking gate:** G4
- **Estimated effort:** 16
- **Done:** [ ]

#### TASK P4.006: G4 decision — record the falsification outcome
- **What:** The G4 gate decision: did any cheap baseline explain the effect?
- **Where:** `notes/decision-gates/g4-cheap-baseline.md`, `docs/GATE_G4_REVIEW.md`,
  `docs/DECISION_LOG.md`
- **Why:** spec §3 · AGENTS.md §3, §7 · hard stop.
- **Inputs:** P4.001, P4.002, P4.003, P4.004, P4.005
- **Acceptance criteria:**
  1. A table compares treatment vs B1–B4 on M against the pre-registered ε, with McNemar p and
     discordant counts for each pair.
  2. An explicit **PASS** (all baselines fail to explain the effect → proceed) or **BLOCK**
     (a baseline explains it → re-scope or terminate) is recorded.
  3. If BLOCK, a negative-results entry is written and the project concludes honestly. This is a
     legitimate and expected outcome, not a failure of execution.
- **Evidence of completion:** gate review + DECISION_LOG entry with human approval.
- **Validation:** human review.
- **Measurements / logs:** full comparison table.
- **Dependencies:** P4.001, P4.002, P4.003, P4.004, P4.005
- **Blocking gate:** G4
- **Estimated effort:** 3
- **Done:** [ ]

---

## Phase 5 — Main Run (Gate G5)

#### TASK P5.001: G5 launch approval for the main full-parameter run
- **What:** Written launch approval with a cost estimate, before the most expensive job.
- **Where:** `docs/GATE_G5_REVIEW.md`, `docs/DECISION_LOG.md`
- **Why:** spec §8 · AGENTS.md §7 · always-present launch gate.
- **Inputs:** P4.006 (PASS)
- **Acceptance criteria:**
  1. Projected wall-clock and cluster-days are computed from P1.006's measured throughput.
  2. The final recipe is frozen and hashed before launch.
  3. Human approval is explicit and dated.
- **Evidence of completion:** `docs/GATE_G5_REVIEW.md`.
- **Validation:** human review.
- **Measurements / logs:** projected hours, frozen config hash.
- **Dependencies:** P4.006
- **Blocking gate:** G5
- **Estimated effort:** 2
- **Done:** [ ]

#### TASK P5.002: Execute the main training run
- **What:** The production fine-tune of Qwen3.8-27B on the full frozen corpus.
- **Where:** `runs/phase5-main-001/`
- **Why:** spec §4 · AGENTS.md §4
- **Inputs:** P5.001
- **Acceptance criteria:**
  1. Progress emitted every ≤100 steps; resume state written each epoch.
  2. Run completes or fails with a recorded trace; either way the experiment log is updated.
  3. Checkpoints are retained per epoch so the shipped epoch can be chosen on evidence.
- **Evidence of completion:** run-id `phase5-main-001`, checkpoints, `metrics.json`.
- **Validation:** loss curve reviewed; a mid-run generation sample read by hand.
- **Measurements / logs:** loss, tok/s, peak memory, ETA, per-epoch checkpoints.
- **Dependencies:** P5.001
- **Blocking gate:** G5
- **Estimated effort:** 48
- **Done:** [ ]

#### TASK P5.003: Select the shipping checkpoint on evidence
- **What:** A recorded choice of which epoch ships, with the comparison that decided it.
- **Where:** `docs/decisions/0006-checkpoint-selection.md`
- **Why:** spec §3 · AGENTS.md §2.6 · "last epoch" is not a justification.
- **Inputs:** P5.002
- **Acceptance criteria:**
  1. Every retained epoch is scored on M.
  2. The selected epoch is justified against the alternatives, including any overfitting signal.
- **Evidence of completion:** `docs/decisions/0006-checkpoint-selection.md`.
- **Validation:** human review of the per-epoch table.
- **Measurements / logs:** metric M per epoch.
- **Dependencies:** P5.002
- **Blocking gate:** G5, G6
- **Estimated effort:** 4
- **Done:** [ ]

#### TASK P5.004: Measure capability retention outside the training domain
- **What:** A forgetting check on capabilities the corpus does not cover.
- **Where:** `runs/phase5-retention-001/`, `docs/EXPERIMENT_LOG.md`
- **Why:** spec §5 · AGENTS.md §2.6 · the model is intended as a daily driver; a model that gains
  on coding and collapses elsewhere is not shippable, and the vision tower was never trained.
- **Inputs:** P5.003
- **Acceptance criteria:**
  1. General-knowledge and instruction-following retention measured against base.
  2. Vision/multimodal capability confirmed non-destroyed, or the loss quantified and disclosed.
  3. Any regression is reported in the model card, not omitted.
- **Evidence of completion:** run-id `phase5-retention-001`, `metrics.json`.
- **Validation:** paired comparison vs `phase3-base-001`.
- **Measurements / logs:** retention deltas per suite.
- **Dependencies:** P5.003
- **Blocking gate:** G6
- **Estimated effort:** 6
- **Done:** [ ]

---

## Phase 6 — Capability Gate (G6)

#### TASK P6.001: Paired comparison — fine-tuned vs base on M
- **What:** The headline comparison, run as a paired test.
- **Where:** `runs/phase6-capability-001/`, `docs/GATE_G6_REVIEW.md`
- **Why:** spec §3 · AGENTS.md §2.7 · this is the claim that justifies publication.
- **Inputs:** P5.003, P3.005
- **Acceptance criteria:**
  1. McNemar p, discordant counts, and the 6-item sensitivity floor are all reported.
  2. Wins are stated per suite, never as a single averaged number.
  3. Before reporting, ≥3 flipped items are read by hand to confirm they are genuine.
- **Evidence of completion:** run-id `phase6-capability-001`, gate review.
- **Validation:** `pytest -k compare`; hand-read of discordant items.
- **Measurements / logs:** per-suite accuracy, discordant items, p-values.
- **Dependencies:** P5.003, P3.005
- **Blocking gate:** G6
- **Estimated effort:** 6
- **Done:** [ ]

#### TASK P6.002: Re-confirm the G4 result at main-run scale
- **What:** Verification that the traces advantage over B1 survives at full scale.
- **Where:** `docs/GATE_G6_REVIEW.md`
- **Why:** spec §3 · AGENTS.md §3 · G4 compared arms at baseline scale; the shipped model is
  bigger-trained, and the control must be re-checked against it.
- **Inputs:** P6.001, P4.002
- **Acceptance criteria:**
  1. Treatment-at-scale vs B1 compared on M against ε.
  2. If the advantage disappears at scale, that is recorded and publication is reconsidered.
- **Evidence of completion:** comparison table in `docs/GATE_G6_REVIEW.md`.
- **Validation:** human review.
- **Measurements / logs:** comparison table.
- **Dependencies:** P6.001, P4.002
- **Blocking gate:** G6
- **Estimated effort:** 3
- **Done:** [ ]

#### TASK P6.003: G6 decision — capability gate
- **What:** Explicit PASS/BLOCK on the publication-justifying claim.
- **Where:** `docs/GATE_G6_REVIEW.md`, `docs/DECISION_LOG.md`
- **Why:** spec §8 · AGENTS.md §7
- **Inputs:** P6.001, P6.002, P5.004
- **Acceptance criteria:**
  1. PASS requires a win on tool calling **and** coding that exceeds the sensitivity floor.
  2. Retention regressions are disclosed in the decision, not deferred.
- **Evidence of completion:** gate review + DECISION_LOG entry with human approval.
- **Validation:** human review.
- **Measurements / logs:** final comparison table.
- **Dependencies:** P6.001, P6.002, P5.004
- **Blocking gate:** G6
- **Estimated effort:** 2
- **Done:** [ ]

---

## Phase 7 — Memorization Audit & Publication (Gate G7)

> Highest-precedence gate. See `AGENTS.md` §11. No weights leave this machine before G7 passes.

#### TASK P7.001: Pre-register the memorization pass criteria
- **What:** The extraction thresholds, written down before the audit runs.
- **Where:** `docs/GATE_G7_REVIEW.md` (criteria section), `docs/DECISION_LOG.md`
- **Why:** spec §5 · AGENTS.md §11.3 · a threshold chosen after seeing results is not a threshold.
- **Inputs:** P6.003
- **Acceptance criteria:**
  1. The n-gram verbatim-reproduction threshold is stated as a number.
  2. The list of private repos to probe is fixed.
  3. Recorded with a date preceding any P7.002 run.
- **Evidence of completion:** DECISION_LOG entry.
- **Validation:** human review; timestamp ordering.
- **Measurements / logs:** n/a
- **Dependencies:** P6.003
- **Blocking gate:** G7
- **Estimated effort:** 2
- **Done:** [ ]

#### TASK P7.002: Run targeted extraction probes for private content
- **What:** An extraction attack against the shipped checkpoint.
- **Where:** `src/suvadu/audit/extract.py`, `runs/phase7-extraction-001/`
- **Why:** spec §5 · AGENTS.md §11.2
- **Inputs:** P7.001
- **Acceptance criteria:**
  1. Probes cover Uyir, TamilLM, Pinnal-Core, ChittiOS, Kickoff-bot (Certinia/Salesforce),
     HouseBuild, career-ops.
  2. Prefix-prompting at non-zero temperature is included, not only greedy decoding.
  3. A secret-pattern sweep runs over all generated output.
  4. Results are reported with the denominator (probes attempted), not as a bare "clean".
- **Evidence of completion:** run-id `phase7-extraction-001`, findings file (local only).
- **Validation:** hand-read of the highest-similarity generations.
- **Measurements / logs:** max verbatim n-gram match per repo, secret hits.
- **Dependencies:** P7.001
- **Blocking gate:** G7
- **Estimated effort:** 10
- **Done:** [ ]

#### TASK P7.003: G7 decision — publication safety
- **What:** Explicit PASS/BLOCK on publishing weights.
- **Where:** `docs/GATE_G7_REVIEW.md`, `docs/DECISION_LOG.md`
- **Why:** spec §5 · AGENTS.md §11 · highest precedence.
- **Inputs:** P7.002, P2.003
- **Acceptance criteria:**
  1. Extraction results are compared against the pre-registered threshold.
  2. Licence determination confirms every training source is redistributable.
  3. On failure, the recorded remedy is corpus repair + retrain — a disclaimer is not an
     acceptable resolution.
- **Evidence of completion:** gate review + DECISION_LOG entry with human approval.
- **Validation:** human review.
- **Measurements / logs:** extraction table.
- **Dependencies:** P7.002, P2.003
- **Blocking gate:** G7
- **Estimated effort:** 3
- **Done:** [ ]

#### TASK P7.004: Write the model card with claim boundaries
- **What:** The Hugging Face model card.
- **Where:** `docs/MODEL_CARD.md`
- **Why:** spec §9 · AGENTS.md §2.7 · every number cites a run ID and is re-read at writing time.
- **Inputs:** P7.003 (PASS)
- **Acceptance criteria:**
  1. Every quantitative claim cites a run ID and states its p-value and sensitivity floor.
  2. A "What was not measured" section exists and is specific.
  3. Training-data provenance and the memorization audit result are disclosed.
  4. Known regressions from P5.004 are stated.
- **Evidence of completion:** `docs/MODEL_CARD.md`.
- **Validation:** every number re-verified against its artifact at writing time, not recalled.
- **Measurements / logs:** n/a
- **Dependencies:** P7.003
- **Blocking gate:** G7
- **Estimated effort:** 4
- **Done:** [ ]

#### TASK P7.005: Publish to Hugging Face
- **What:** The published model repo.
- **Where:** `Murai-Labs/Suvadu-Qwen3.8-27B` on Hugging Face
- **Why:** spec §9 · AGENTS.md §11.1
- **Inputs:** P7.004
- **Acceptance criteria:**
  1. G7 recorded as PASS with human approval before any upload begins.
  2. Uploaded files are enumerated and reviewed — no optimizer states, no corpus, no logs
     containing paths from the private machine.
  3. The model card matches `docs/MODEL_CARD.md` exactly.
- **Evidence of completion:** HF repo URL + commit SHA.
- **Validation:** post-publish file listing reviewed by a human.
- **Measurements / logs:** n/a
- **Dependencies:** P7.004
- **Blocking gate:** G7
- **Estimated effort:** 2
- **Done:** [ ]

---

## Deferred / Out of Scope

Recorded so they are not silently forgotten:

- **NVFP4 re-quantization and fp8-KV recalibration.** The 107.6 tok/s serving recipe in
  `Murai-Labs/qwen3.8-27b-rtx5090` depends on calibration scales shipped with that checkpoint. A
  fine-tuned model does not inherit them. Serving fast is a separate project.
- **MTP head re-alignment.** The in-checkpoint `model_mtp.safetensors` drafts for the *base*
  model. After fine-tuning, acceptance rate will drift and the ~2× speedup degrades. Quality
  should be unaffected (rejections resample), but this is unverified.
- **Tamil capability.** Not in the mixture; TamilLM data is a separate axis that would compete
  for capacity.
- **Vision tower training.** Never trained here — only checked for non-destruction (P5.004).
