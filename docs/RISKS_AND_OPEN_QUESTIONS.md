# Suvadu — Risks and Open Questions

## Format

### Q<NNN> — <Title>

**Status:** <Open / Resolved YYYY-MM-DD>
**Blocking Impact:** <Low / Medium / High>
**Needed For:** <what cannot proceed until this is resolved>
**Resolution Path:** <how to resolve it>

---

### Q001 — Cheap baselines have not been run

**Status:** Open
**Blocking Impact:** High
**Needed For:** G4 approval; everything past Phase 4; publication.
**Resolution Path:** Execute B1–B4 on metric M (TASKs P4.001–P4.005) and record the comparison
against the pre-registered ε = 3.0 pp. B2 (prompt-only/RAG) is cheapest and runs first — if
prompting alone matches the target, the project can conclude before any GPU-hours are spent
on SFT.

---

### Q002 — The training toolchain is unverified on aarch64 + sm_121

**Status:** **Resolved 2026-08-16** — affirmatively.
**Blocking Impact:** was High
**Needed For:** G1; the entire project's feasibility as specified.
**Resolution:** TASK P1.001 ran on `spark-1003`. Inside `nvcr.io/nvidia/pytorch:25.11-py3`:
`torch 2.10.0a0+nv25.11`, `capability (12, 1)` = sm_121 on GB10, `transformers 5.15.0` and
`accelerate 1.14.0` installed from PyPI on aarch64 **with no build step and no missing wheels**,
and `AutoConfig.from_pretrained("Qwen/Qwen3.8-27B")` resolved `model_type: qwen3_5` without
`trust_remote_code`. The project does not re-scope to LoRA on toolchain grounds. Full output in
`configs/phase1/env-report.md`; decision in `docs/decisions/0001-training-stack.md`.

**Residual:** this says the toolchain *imports*, not that a 27B step *fits*. That is Q006/P1.003.
It also used runtime `pip install`, which is a provenance hole now tracked as TASK P1.007.

---

### Q003 — Private source code in the corpus is a publication blocker, not a caveat

**Status:** Open
**Blocking Impact:** High
**Needed For:** G7; any push of weights to Hugging Face.
**Resolution Path:** TASKs P2.002 and P7.002. The 2026-07-13 scrub caught secrets and PII but
explicitly did **not** remove private source code pasted into tool outputs. Full-parameter
fine-tuning memorizes. Repos at risk include unpublished research (Uyir, TamilLM, Pinnal-Core),
a private repo (ChittiOS), and client-adjacent work (Kickoff-bot, Certinia/Salesforce). Remedy on
failure is corpus repair and retraining — never a disclaimer.

---

### Q004 — Evaluation contamination from the traces themselves

**Status:** Open
**Blocking Impact:** High
**Needed For:** G2, G6; the validity of the entire capability claim.
**Resolution Path:** TASK P2.006. The corpus is real coding sessions on a machine where
benchmark problems have plausibly been discussed. If HumanEval or MBPP items appear verbatim in
training data, the headline result is meaningless. Decontamination is a gate, and removals must
be hand-checked so false positives are not mistaken for a clean sweep.

---

### Q005 — Third-party datasets display no licence

**Status:** Open — **narrowed 2026-08-16**, no longer blocks training.
**Blocking Impact:** Low for training; **High for publication (G7)**.
**Needed For:** G7.
**Resolution Path:** Checked 2026-08-16: `11-47/claude_opus_4.8_distill_5k` states Apache-2.0
(5,000 rows, instruction/response with `<think>` traces). The seven `r0b0tlab` datasets display
**no licence at all**.

Ramchand **approved their use on 2026-08-16** (DEC-0008), on the basis that the author is a
friend who omitted the licence file rather than withheld it. That is his call and it is made;
training proceeds with them included.

What remains open is narrower and belongs to G7: a personal assurance is not an artifact. Before
publication, TASK P2.003 must obtain **one written confirmation** — a licence file added to the
dataset repos, or a message granting redistribution — and record which. Without it, a model card
claiming redistributable training data would be asserting something no document supports.

Separately unresolved and **not** addressed by the author's permission: these datasets are
distillations of other vendors' model outputs, which is a terms-of-service question about those
vendors, not a licensing question about the datasets.

---

### Q006 — The full-FT memory figure is arithmetic, not a measurement

**Status:** Open — **inputs refined 2026-08-16**, conclusion still unmeasured.
**Blocking Impact:** Medium
**Needed For:** G1; the feasibility of full-parameter FT across 2 nodes.
**Resolution Path:** TASK P1.003.

The weight term is no longer an estimate. Read from `model.safetensors.index.json` on
2026-08-16: **`total_size` = 55,562,855,904 bytes = 51.75 GiB, ~27.78 B params**, 1199 tensors
across 18 shards. The earlier "54 GB" was close but was arithmetic from a rounded parameter
count.

Revised estimate: 51.75 (weights, bf16) + 51.75 (gradients, bf16) + ~51.75 (8-bit Adam, two
states at 1 byte/param) ≈ **155 GiB**, before activations, gradient-checkpointing buffers, FSDP
shard overhead, and the hybrid model's recurrent state. Still exceeds one node (121 GB ≈ 112.7
GiB); still requires 2-node FSDP.

**This remains a computation, not a profile, and must not be cited as measured.** Two levers
discovered in P1.001 could change it materially and are untested: the checkpoint carries **333
vision-tower tensors** and **13 MTP tensors** that a text-only SFT has no reason to train, and
the config exposes a `language_model_only` flag. Freezing or excluding those removes them from
the gradient and optimizer budget. If the real profile does not fit, the README is corrected
rather than the plan quietly changed.

---

### Q007 — Train/serve formatting skew around reasoning tokens

**Status:** Open
**Blocking Impact:** Medium
**Needed For:** G2; the validity of any eval result.
**Resolution Path:** TASK P2.005. Qwen3.8's chat template treats `reasoning_content` specially,
and `preserve_thinking` changes rendering — the companion inference repo found the stock template
emits an empty `<think></think>` per prior assistant turn unless `preserve_thinking: false` is
sent. Meanwhile the external datasets carry `<think>` blocks inline in the response text. If
training-time and inference-time rendering differ, measured gains may be artifacts of formatting.

---

### Q008 — Fine-tuning invalidates the measured serving recipe

**Status:** Open (deferred, out of scope)
**Blocking Impact:** Low for this project; High for daily-driver use
**Needed For:** Nothing in G0–G7. Recorded so it is not discovered later as a surprise.
**Resolution Path:** Separate project. The 107.6 tok/s recipe depends on NVFP4 weights plus the
**fp8 KV calibration scales shipped in that checkpoint**; a fine-tuned model does not inherit
them and must be re-quantized and re-calibrated. Separately, the in-checkpoint MTP head drafts
for the *base* model — after fine-tuning, acceptance rate drifts and the ~2× speedup degrades,
though quality should hold since rejections resample.

---

### Q009 — Only 16 of 64 layers are attention

**Status:** Open
**Blocking Impact:** Medium
**Needed For:** G4 (baseline B4 target-module choice); interpretation of any LoRA result.
**Resolution Path:** TASK P4.004. Qwen3.8-27B is `model_type: qwen3_5`, a hybrid of
16 × [3 × Gated DeltaNet → 1 × Gated Attention]. A default LoRA preset targeting `q/k/v/o` and
`gate/up/down` does not mean the same thing here as on a standard dense transformer — 48 of 64
layers are recurrent. Target modules must be chosen deliberately and the choice justified, or the
B4 comparison is not interpretable.
