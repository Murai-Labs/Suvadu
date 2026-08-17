# DEC-D0007 — Freeze the vision tower by default; make the MTP head a policy choice

Date: 2026-08-16
Task: P1.004
Gate: G1
Status: Decided as the default; the choice is a config field, not a hardcoded constant

## Context

Qwen3.8-27B is `Qwen3_5ForConditionalGeneration` — one checkpoint containing three things.
Measured from `model.safetensors.index.json` on 2026-08-16:

| Component | Tensors |
|---|---|
| `model.language_model.*` | 850 |
| `model.visual.*` | **333** |
| `mtp.*` | 13 |
| `lm_head` | 1 |

Suvadu's corpus is text. Every trainable parameter costs a gradient plus optimizer state, and
the 2-node fit is already marginal: ~155 GiB estimated against 2 × ~112.7 GiB usable.

## Decision

Default policy **`vision_and_mtp`** — train the language model and `lm_head`, freeze the vision
tower and the MTP head. Implemented in `src/suvadu/train/freeze.py`; selected with
`--freeze-policy`; recorded in the run manifest and in `freeze_summary.json` per run.

Three policies exist so each arm can name what it did:

- `vision_and_mtp` — default; text-only SFT.
- `vision_only` — freeze the vision tower, keep the MTP head trainable.
- `none` — train everything. Included so the "no freezing" arm is *nameable and comparable*,
  not because it is expected to fit.

## Rationale

**Vision tower.** A text corpus provides no gradient signal for it that is worth having.
Training it spends scarce memory to drift a capability the project explicitly does not measure —
and P5.004 has to verify that capability was not destroyed, which freezing makes trivially true
rather than something to hope for.

**MTP head — the genuinely two-sided one.** The head ships in the checkpoint and drafts tokens
for speculative decoding; it is where the companion inference repo's measured ~2× decode speedup
comes from (107.6 vs 51.5 tok/s), and that speedup was *lossless* across 200 paired GSM8K items
with zero discordant results.

It drafts for the **base** model. Fine-tune the target and freeze the head, and the draft
distribution no longer matches what it is drafting for: acceptance rate falls and the speedup
degrades. Quality should hold, because rejected drafts are resampled from the target — that is
what made the original result lossless in the first place.

Training the head on the corpus keeps it aligned, but adds 13 tensors of gradient and optimizer
state to a marginal fit, and the corpus was never designed to train a draft head.

Neither option is free, so this is a config field. The default freezes it because **serving speed
is explicitly out of scope** for this project (NVFP4 re-quantization and fp8-KV recalibration are
already deferred), and a marginal memory fit is a live constraint today.

## Alternatives considered

- **Train everything (`none`).** Rejected as a default: 346 tensors of gradient and optimizer
  state bought with memory the fit does not have, for capabilities the metric does not measure.
- **Hardcode the freeze.** Rejected: an unexamined default is exactly what Q009 warns about on
  the LoRA side, and the same reasoning applies here.
- **Use the config's `language_model_only` flag.** Not rejected — untested. If it loads a
  text-only model outright, it may be strictly better than freezing, since frozen parameters
  still occupy weight memory even when they carry no gradient. Worth testing in P1.003.

## Consequences

1. **P5.004's vision-retention check becomes cheap.** With the tower frozen, "did we destroy
   it?" has an a-priori answer, and the check confirms rather than discovers.
2. **A published model card must state that the MTP head was not updated**, and that the
   speculative-decoding speedup measured on the base model does not transfer unchanged.
3. **Frozen ≠ absent.** Frozen parameters still occupy weight memory. This decision reduces
   gradient and optimizer state, not the 51.75 GiB of weights. P1.003 measures the real saving.

## Verification

`tests/test_freeze.py` — **11 tests, all passing** (suite total 82). Parameter names are real prefixes read from
the checkpoint index, not invented examples. One test specifically guards Q009: the 48
`linear_attention` layers expose `in_proj_a/b/qkv/z` and `out_proj`, and a classifier keyed on
conventional `q_proj/k_proj/v_proj` names would silently drop them into `OTHER` and freeze them.
