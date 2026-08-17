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

**Status:** **RESOLVED 2026-08-17.** Full-parameter training **fits** on 2 nodes at seq_len 2048.

### Final answer (run `fsdp6-torchao`, both nodes, exit 0)

```
load          366.5s   alloc = 50.97 GiB
shard           5.6s   alloc = 25.50 GiB
optim_init      3.5s   alloc = 25.50 GiB
step1_forward   7.0s   alloc = 34.52 GiB   (+9.02)
step1_backward 32.0s   alloc = 51.55 GiB
step1_optim    98.5s   alloc = 77.06 GiB   (8-bit state materialises here)
step2          36.0s
step3          44.7s
PEAK ALLOCATED = 88.83 GiB    ceiling 111.95    device 121.69
```

**It fits, with ~23 GiB of allocator headroom.** Config: `AutoModelForImageTextToText`, bf16,
vision frozen, gradient checkpointing **actually engaged**, `torchao.optim.AdamW8bit`, FSDP2
across 2 nodes, batch 1 × seq 2048 per rank.

⚠️ **But the system ran close to the edge that torch cannot see.** During steps 2–3,
`MemAvailable` fell to **3.95 and 3.68 GiB** while torch reported only 55.05 GiB allocated —
so roughly 60 GiB was held outside the allocator (CUDA context, NCCL buffers, reserved-but-
unallocated pool). The torch ceiling is necessary and **not sufficient**; `/proc/meminfo` is the
binding constraint on this hardware. Raising `--max-mem-fraction` further would be unsafe
despite the apparent 23 GiB of allocator headroom.

**Still to resolve (moved to Q011):** throughput. Steady-state step is **40.39 s for 4096 global
tokens ≈ 101 tokens/s**, which has severe schedule consequences.

### How this was reached — the single-node result that forced 2-node sharding

`AutoModelForCausalLM`, bf16, seq_len 2048, batch 1. **Correction 2026-08-17:** this run was
recorded as "gradient checkpointing on", and the flag was set — but `memprobe.py` shares the
`model.train()` omission found later in `fsdp_probe.py`, so checkpointing **did not engage**.
The numbers below are therefore *without* effective checkpointing.

| Stage | Result | torch allocated |
|---|---|---|
| `weights` | ok | **50.10 GiB** (26,895,998,464 params × 2 B — exact) |
| `grads` | **OOM** | **116.05 GiB** at a 121.69 GiB device capacity |

**Full-parameter training does not fit on one node.** That is now a measurement, not arithmetic.

Note the gap: weights + bf16 gradients predict 100.2 GiB, and the probe reached **116.05** before
dying — about **16 GiB** of activations and workspace. Originally written up as "even with
gradient checkpointing enabled"; that was wrong, since checkpointing was not engaging (see the
correction above). The 2-node run later measured the checkpointed forward at **+9.02 GiB**.

**The single-node conclusion is unaffected and does not depend on this.** Weights (50.10) +
gradients (50.10) + 8-bit optimizer state (~50.10) = **~150 GiB against a 121.69 GiB device**.
Full-parameter training cannot fit on one node whatever the activation term does.

Also established: on GB10 `torch` reports the device capacity as **121.69 GiB**, i.e. the whole
unified pool. There is no separate VRAM, so a CUDA OOM on this hardware *is* system-memory
exhaustion — see the incident in `notes/integrity-gaps.md` for why that matters operationally.

*(At the time this single-node section was written, the 2-node case and the optimizer-state
cost were both unmeasured. Both are now answered above: it fits, and 8-bit optimizer state
materialises during the first `.step()`, taking peak allocation from 51.55 to 77.06 GiB.)*

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

---

### Q010 — `adamw_8bit` assumes bitsandbytes works on aarch64 + sm_121

**Status:** **Resolved 2026-08-17** — affirmatively.
**Blocking Impact:** was High. The entire 2-node memory argument depended on it.
**Needed For:** G1 (P1.003), and the feasibility of full-parameter FT at all.
**Resolution:** Tested on `spark-1003` inside `suvadu-train:2026-08-17`:

```
bnb_version: 0.50.1
bnb_adamw8bit_step: OK      # AdamW8bit .step() on a cuda bf16 parameter, GB10 / sm_121
torch_fused_adamw: OK       # torch.optim.AdamW(fused=True), the fallback
```

`bitsandbytes` installs from PyPI on aarch64 with no build step, and an 8-bit optimizer step
executes on GB10. The fallback path is also live, so this is de-risked twice over.

**Residual:** this shows the optimizer *initializes and steps on one small tensor*. It does not
show that 8-bit state for ~27.4 B trainable parameters fits, which is the `optim` stage of
P1.003. Keep `optimizer: adamw_8bit` explicit in configs so the assumption stays greppable.

### ⚠️ REOPENED AND RE-RESOLVED 2026-08-17 — the residual above was the real issue

The residual was framed as a question of **scale**. It was actually a question of **tensor type**,
and the distinction mattered:

```
RuntimeError: bitsandbytes.optimizer_update_8bit_blockwise.default:
got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor
before calling distributed operators!
```

**`bitsandbytes` AdamW8bit cannot be used with FSDP2 at all.** Its custom CUDA op does not
dispatch over `DTensor`, and FSDP2 hands the optimizer DTensor-sharded parameters. The check
that closed this gate used a plain `torch.nn.Parameter`, so it proved the kernel runs — not that
it runs on what FSDP2 actually produces. The gate read green on evidence that did not cover the
use.

**Resolution:** switched to **`torchao.optim.AdamW8bit`**, which is DTensor-aware. torchao
0.14.0 already ships inside `nvcr.io/nvidia/pytorch:25.11-py3`, so this additionally removes the
runtime `pip install bitsandbytes` and makes the probe reproducible from the pinned image alone.
`torchao.optim.AdamW4bit` is available as a fallback if 8-bit state proves too large.

**Standing lesson:** a compatibility check must exercise the *same object type* the real code
path produces. "It works on a tensor" does not establish "it works on a sharded tensor".

The ~155 GiB estimate assumes optimizer state of ~1 byte per parameter per moment — i.e. 8-bit
Adam. Standard fp32 AdamW would be **4 bytes × 2 moments = ~222 GiB of optimizer state alone**,
which does not fit across both nodes under any arrangement. So the 8-bit optimizer is not a
tuning preference here; it is load-bearing.

`bitsandbytes` is the usual provider and its aarch64 + Blackwell support is exactly the kind of
thing the standing "wheels are spotty" rule warns about. It was **not** installed or tested in
P1.001, and is deliberately not pinned in `docker/Dockerfile.train` for that reason.

Fallbacks if it does not work, in rough order of preference: torch's fused AdamW with bf16
optimizer states; Adafactor (no second moment, much smaller state); optimizer-state CPU offload
via FSDP; and finally re-scoping to LoRA, where the optimizer state is negligible because
almost nothing is trainable.

`configs/phase1/smoke.yaml` sets `optimizer: adamw_8bit`, so this assumption is at least
explicit and greppable rather than buried in a framework default.

---

### Q011 — Measured throughput makes full-parameter training impractical at corpus scale

**Status:** Open — **the central planning risk of the project.** Raised 2026-08-17 the moment
P1.003 produced a real number.
**Blocking Impact:** **High.** Affects G4 arm design, G5 launch approval, and whether the
project's chosen regime is viable at all.
**Needed For:** G5; realistically also G4, since G4 requires four training arms plus a treatment.

**Measured** (run `fsdp6-torchao`, 2 nodes, batch 1 x seq 2048 per rank):

| | |
|---|---|
| steady-state step | **40.39 s** (first step 98.6 s — includes optimizer-state allocation) |
| tokens per step, global | **4,096** |
| **throughput** | **~101 tokens/s** |
| implied per day | **~8.8 M tokens/day** |

**Computed consequences** (arithmetic from the measurement, not themselves measured):

| Corpus | 1 epoch | 3 epochs |
|---|---|---|
| 50 M tokens | ~5.7 days | ~17 days |
| 100 M tokens | ~11.4 days | ~34 days |
| 200 M tokens | ~23 days | ~68 days |

The corpus token count is not yet known (P2.001), but the July export was ~202k messages, so
100 M+ tokens is the plausible range. **At 3 epochs that is over a month of exclusive
cluster time for a single arm** — and G4 requires B1, B3, B4 and a treatment arm before the main
run is even authorised.

**Levers, in rough order of expected value:**

1. **More tokens per step.** Peak was 88.83 GiB against a 111.95 GiB ceiling. Communication cost
   per step is roughly fixed (~150 GiB of FSDP traffic at a measured 11.55 GB/s ~= 13 s), so
   doubling tokens per step should improve throughput substantially rather than linearly costing
   time. Constrained by the `MemAvailable` headroom above, not by the allocator ceiling.
2. **Fewer epochs.** 3 is a convention here, not a requirement derived from anything.
3. **Shorter G4 arms.** Already the plan — the baselines must be matched to each other, not
   full-length.
4. **Re-scope the regime to LoRA.** Baseline B4 exists precisely to test whether full-parameter
   training earns its cost. If it does not, B4 becomes the recipe and the schedule collapses by
   an order of magnitude.

**Resolution path:** re-run the probe at larger tokens-per-step to find the throughput knee, then
size the corpus and epoch count against a stated cluster-time budget before G5.
