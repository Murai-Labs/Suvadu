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

**Status:** Open
**Blocking Impact:** High
**Needed For:** G1; the entire project's feasibility as specified.
**Resolution Path:** TASK P1.001. Unsloth v0.1.800-beta advertises Qwen3.8-27B fine-tuning and
Unsloth docs state transformers v5 is required — but neither has been verified on GB10. The
standing lab rule is that aarch64+Blackwell wheels are spotty and ML there is container-first.
If the stack does not exist, the project re-scopes to LoRA (baseline B4 becomes the treatment)
and that re-scope is recorded, not silently absorbed.

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

**Status:** Open
**Blocking Impact:** Medium (High if publication proceeds)
**Needed For:** G2, G7.
**Resolution Path:** TASK P2.003. Checked 2026-08-16: `11-47/claude_opus_4.8_distill_5k` states
Apache-2.0 (5,000 rows, instruction/response with `<think>` traces). The seven `r0b0tlab`
datasets display **no licence at all**, and are distillations of other vendors' model outputs
(`qwen3.8-max-glm5.2-kimi-k3`, `deepseek-v4-pro-agentic`, Hermes traces). Unlicensed datasets are
EXCLUDED by default; including one requires an explicit recorded human decision.

---

### Q006 — The 162 GB memory figure is arithmetic, not a measurement

**Status:** Open
**Blocking Impact:** Medium
**Needed For:** G1; the feasibility of full-parameter FT across 2 nodes.
**Resolution Path:** TASK P1.003. The estimate (54 GB weights + 54 GB gradients + 54 GB 8-bit
optimizer state) is computed from parameter counts and ignores activations, FSDP shard overhead,
and the hybrid architecture's recurrent state. It must not be cited as measured. If the real
profile does not fit, the README is corrected rather than the plan quietly changed.

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
