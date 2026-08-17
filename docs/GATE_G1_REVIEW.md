# Gate G1 Review — Training Stack Verified

## Purpose

G1 proves a 27B full-parameter optimizer step is physically possible on this cluster BEFORE days are spent building a corpus. If the toolchain does not exist for aarch64 + sm_121, the project re-scopes to LoRA, and that must be discovered now rather than after the data work.

## Gate Status

Status: **Not yet approved.** Approval is recorded in `docs/DECISION_LOG.md`; this file
preserves the review evidence.

## Required Evidence

| Requirement | Evidence (path) | Status |
|-------------|-----------------|--------|
| Toolchain report, transformers v5 + sm_121 confirmed | `configs/phase1/env-report.md` | **PASS 2026-08-16** |
| Training-stack decision recorded | `docs/decisions/0001-training-stack.md` | **Present** |
| Pinned training image (no runtime pip) | `suvadu-train:<tag>` | TASK P1.007 |
| BF16 base resident on **both** nodes, revision pinned | `configs/phase1/base-model.json` | **COMPLETE 2026-08-17** |
| Measured memory profile, full FT and LoRA | `runs/phase1-memprobe-001/metrics.json` | TASK P1.003 |
| Training entrypoint with progress + provenance | `src/suvadu/cli/train.py` | TASK P1.004 |
| DeepSeek restore procedure captured BEFORE shutdown | `docs/RUNBOOK.md restore section` | TASK P1.005 |
| Smoke run >=50 steps, loss decreasing | `runs/phase1-smoke-001/` | TASK P1.006 |
| Measured throughput and projected main-run wall-clock | `runs/phase1-smoke-001/metrics.json` | TASK P1.006 |

Rows not marked PASS/Present are **Missing** until their task completes and the artifact is
re-read at the moment this table is filled in — not recalled from the session that produced it.

## Progress — P1.001 result (2026-08-16)

The largest open risk on this project (Q002: does a training toolchain exist for aarch64 +
sm_121?) is **resolved in the affirmative**:

```
torch 2.10.0a0+b558c986e8.nv25.11    capability (12, 1) = sm_121    device NVIDIA GB10
transformers 5.15.0                  accelerate 1.14.0
AutoConfig("Qwen/Qwen3.8-27B") -> model_type: qwen3_5
```

Inside `nvcr.io/nvidia/pytorch:25.11-py3`. `transformers` and `accelerate` installed from PyPI
on aarch64 with no build step — the wheel-availability risk did not materialize. The project
does **not** re-scope to LoRA on toolchain grounds.

Also measured from `model.safetensors.index.json` (no weights downloaded): the BF16 checkpoint is
**51.75 GiB / ~27.78 B params**, 1199 tensors in 18 shards, comprising 850 language-model
tensors, **333 vision-tower tensors** and 13 MTP tensors. The 48 linear-attention / 16
full-attention layer split is now confirmed from the model's own config.

## Progress — P1.002 result (2026-08-17)

BF16 base resident on **both** nodes: 55,586,114,863 bytes, 32 files, 18 shards, byte-identical
across nodes, `exit=0`, ~84 min each. Revision pinned. Config and tokenizer load from the local
directory on both nodes; 18/18 shards present, 1199 tensors indexed.

Every byte reconciled against the index's `total_size` (delta = safetensors headers + tokenizer
files; **zero unexplained bytes**).

**Integrity caveat worth carrying forward:** upstream's `crc32.txt` fails on 3 of 8 files. It is
stale, not a corrupt download — the failing set is exactly the files replaced after the manifest
was written, and the failures are byte-identical across two independent downloads. It also does
not cover the weight shards at all, so it provides no assurance about the 51.75 GiB that matters.

## Explicit Non-Results

- No corpus exists.
- No evaluation has been run.
- **The weight shards themselves have not been cryptographically verified.** No upstream digest
  covers them. Confidence rests on: two independent downloads agreeing byte-for-byte, HTTP-level
  size/etag checks inside `snapshot_download`, and a complete structural match against the index.
  That is reasonable, and it is not a checksum.
- **No model has been loaded into memory.** Config and tokenizer only — the 51.75 GiB of weights
  have never been instantiated, so nothing yet shows they load, shard, or fit.
- **No training step has run and no memory has been measured.** P1.001 proves the toolchain
  imports and the model resolves; it proves nothing about whether an optimizer step fits.
- The full-FT memory figure — now ~155 GiB rather than the earlier 162 GB — is still
  **arithmetic**. Q006 stays open until P1.003 measures it.
- The probe touched **node 1 only**. Multi-node FSDP and the 200G fabric remain untested.
- Nothing is version-pinned yet; the probe used runtime `pip install` (P1.007).

## Required Next Work Before Approval

Complete every task named in the evidence table, then fill this table from the artifacts.

## Approval Template (paste into docs/DECISION_LOG.md when granted)

```
## DEC-XXXX — G1 approved
Date: <YYYY-MM-DD>
Task/Gate: G1
Decision: G1 (Training Stack Verified) approved; next phase unblocked.
Rationale: <evidence summary>.
Evidence / Source Docs: docs/GATE_G1_REVIEW.md, <run ids>.
Human Approval: <name> on <date>.
```
