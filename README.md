# Suvadu

*சுவடு — "trace, footprint, the mark left behind."*

Does fine-tuning a 27B model on your own agent-session traces actually make it better at
agentic work — or does it just make it sound like you?

Suvadu is a controlled attempt to answer that on **Qwen3.8-27B**, full-parameter, on a 2-node
DGX Spark cluster. The headline is not a model. The headline is whether the model survives a
control that most personal-fine-tune projects never run.

## Current Phase

**G0 — skeleton complete.** No training has been run. No data has been built. No claim in this
repo is yet backed by a measurement.

## Research Thesis

- **Claim 1:** Full-parameter SFT of Qwen3.8-27B on personal agent traces plus curated agentic
  datasets beats base Qwen3.8-27B on tool calling (BFCL) and coding (EvalPlus), under a paired
  McNemar test.
- **Claim 2:** That gain is attributable to the *personal traces specifically* — it survives a
  token-matched control trained on public data alone.

**Null hypothesis the project must defeat:**

> The traces-trained model is hypothesized to outperform baselines B1–B4 on metric M. If any of
> B1–B4 match it within **ε = 3.0 percentage points** on M, the hypothesis is falsified and the
> project concludes without publication.

Claim 2 is the one that matters. A personal fine-tune that beats base but *not* the public-data
control has demonstrated that SFT works, which is already known — it has demonstrated nothing
about the traces.

### The baselines that can kill this project

| | Baseline | What it would prove |
|---|---|---|
| **B1** | Token-matched public-mixture-only SFT | The traces added nothing beyond ordinary data |
| **B2** | Prompt-only / RAG, zero training | No training was needed at all |
| **B3** | Traces at 10% subsample | Corpus scale is not the active ingredient |
| **B4** | LoRA instead of full FT | Full-parameter training didn't earn its compute |

ε is not a guess. It is the **measured sensitivity floor** of the paired McNemar harness reused
from [`Murai-Labs/qwen3.8-27b-rtx5090`](https://github.com/Murai-Labs/qwen3.8-27b-rtx5090) — 6
discordant items, independent of n, which at n=200 is exactly 3.0 pp. Differences smaller than
that are invisible to the test, so claiming them would be dishonest.

## Deliverables

- A published Hugging Face model — **only if G6 and G7 both pass**.
- A model card with explicit claim boundaries, including what was *not* measured.
- This repo: reproducible training and evaluation code, plus the full decision and gate record.
- A technical report.

## Why this repo may end with no model

Two gates can terminate it, and both are designed to:

**G4 — cheap-baseline falsification.** The Murai Labs standing rule, written after two projects
(UYIR, MARMAM) each spent months on effects that the dumbest available alternative explained
just as well. Roughly six months of cumulative cost bought this gate. It is not a formality.

**G7 — memorization audit.** The corpus is built from real sessions on a personal machine and
contains private source code that the scrubbing pass explicitly did not remove. Full-parameter
fine-tuning memorizes; publishing weights publishes what was memorized. No weights are pushed
before this audit passes, and a failure is repaired by fixing the corpus and retraining — never
by adding a disclaimer.

## Hardware

2× NVIDIA GB10 (DGX Spark), 121 GB unified memory each, 200G ConnectX-7 fabric measured at
111 Gb/s RDMA / 13.6 GB/s NCCL all-reduce. Full-parameter FSDP across both nodes.

Sizing note: Qwen3.8-27B in BF16 is ~54 GB of weights. With gradients and 8-bit optimizer state
that is ~162 GB — more than one node holds, which is why this is a 2-node job and why LoRA
(~65–75 GB, single node) exists as baseline B4 rather than as the default.

## Repo Guide

| File | Purpose |
|------|---------|
| `AGENTS.md` / `CLAUDE.md` | Operating contract (byte-identical) |
| `CODEX.md` | Codex pointer to AGENTS.md |
| `tasks/atomic-task-list.md` | Canonical atomic task list |
| `docs/` | Decision log, experiment log, reproducibility, gate reviews |
| `configs/` | Locked, versioned configs |
| `runs/` | Run manifests and metrics (weights gitignored) |
| `src/suvadu/` | Package source |

`STATUS.md`, `CHECKPOINT.md` and `notes/` are intentionally **not** in this public repo — per
Murai Labs convention, internal trackers and the append-only audit trail stay local.

## Data posture

The repo is text-free. No trace-derived data, no merged corpus, and no weights are ever
committed. Corpus provenance travels as hashes and manifests only.

## Licence

Code: MIT. Model weights, when and if published, carry their own licence and attribution;
Qwen3.8-27B is Apache-2.0.

## Status of every number in this README

There are none yet. That is deliberate — at G0 the honest count of measurements is zero.
