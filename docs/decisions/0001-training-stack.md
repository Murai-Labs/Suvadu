# DEC-D0001 — Training stack: NGC PyTorch container + transformers v5

Date: 2026-08-16
Task: P1.001
Gate: G1
Status: Decided (probe passed); image pinning outstanding (P1.007)

## Decision

Train inside **`nvcr.io/nvidia/pytorch:25.11-py3`** on the DGX Spark nodes, with
`transformers>=5` and `accelerate` layered on top, rather than on the host or in one of the
resident vLLM images.

## Rationale

Verified on `spark-1003`, 2026-08-16 — full output in `configs/phase1/env-report.md`:

- The **host has no torch and no transformers**, and no `uv`/`hf` on PATH. Training on the host
  would mean building a CUDA-13 aarch64 stack from scratch, which is exactly the "aarch64 +
  Blackwell wheels are spotty" trap the standing rule warns about.
- The NGC image already carries `torch 2.10.0a0+b558c986e8.nv25.11` built for this hardware, and
  reports `capability (12, 1)` = **sm_121** on GB10. That is the single hardest thing to get
  right on this platform, and it is already correct in the image.
- `transformers 5.15.0` and `accelerate 1.14.0` installed from PyPI on aarch64 with **no build
  step and no missing wheels** — the risk that motivated Q002 did not materialize.
- `AutoConfig.from_pretrained("Qwen/Qwen3.8-27B")` resolves to `model_type: qwen3_5`,
  arch `Qwen3_5ForConditionalGeneration`, without `trust_remote_code`.

## Alternatives considered

- **The resident vLLM images** (`vllm-node-main`, `eugr/spark-vllm-b12x`, etc., six of them,
  ~23 GB each). Rejected: they are inference runtimes assembled around vLLM and its patches.
  Reusing one would couple this project's training environment to the serving stack's patch
  history for no benefit. One of them is also actively serving Gemma right now.
- **Unsloth**, which advertises Qwen3.8-27B fine-tuning in v0.1.800-beta. Not rejected — not yet
  evaluated. Unsloth's advantage is memory efficiency on a *single* GPU, and this project's
  binding constraint is a 2-node FSDP shard. It remains the obvious candidate for baseline B4
  (LoRA, single node), where its strengths actually apply. Deferred to P4.004.
- **Host-native build.** Rejected as above.

## Consequences

1. **The image must be pinned before any citable run.** The probe used `pip install` at
   container start, which resolves whatever PyPI serves that day. Two runs a week apart would
   report the same "environment" while differing in library version. TASK **P1.007** builds and
   tags `suvadu-train:<date>` with a lockfile.
2. **Freezing the vision tower and MTP head is now a live design question.** The BF16 checkpoint
   carries 333 vision tensors and 13 MTP tensors alongside 850 language-model tensors. A
   text-only SFT that trains all of them wastes gradient and optimizer memory on parameters the
   corpus never exercises — and the 2-node fit is marginal enough that this matters. P1.004 must
   decide explicitly rather than inherit a framework default. The config exposes a
   `language_model_only` flag that may make this trivial; untested.
3. **LoRA target modules cannot be written from habit.** The 48 recurrent layers expose
   `in_proj_a / in_proj_b / in_proj_qkv / in_proj_z / out_proj`, not `q_proj/k_proj/v_proj`. A
   conventional preset silently misses three quarters of the network. This is now recorded
   evidence for Q009 rather than an inference from the companion repo's README.

## Evidence

`configs/phase1/env-report.md` — pasted output of every command, run 2026-08-16 on spark-1003.

## What this does NOT establish

That a 27B optimizer step fits. No weights were downloaded, no step was run, no memory was
measured, and the probe touched one node only. The 155 GiB full-FT figure remains arithmetic
(Q006), and multi-node FSDP over the 200G fabric is entirely untested.
