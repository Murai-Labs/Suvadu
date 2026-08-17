# Phase 1 Environment Report — TASK P1.001

Executed 2026-08-16 on `spark-1003` (Murailabs-Spark). Every line below is pasted command
output, not recollection.

## Verdict

**PASS.** A transformers-v5 stack resolves Qwen3.8-27B and sees sm_121 on GB10. The project does
**not** need to re-scope to LoRA on toolchain grounds.

One caveat that is not a pass/fail issue but is a reproducibility issue: `transformers` was
installed at container runtime, not baked into an image. That is fine for a probe and **not**
acceptable for a run whose results get cited. See "Required follow-up".

## Host

```
Linux spark-1003 6.17.0-1029-nvidia aarch64 GNU/Linux
Python 3.12.3        (host — no torch, no transformers installed)
NVIDIA GB10, driver 580.173.02
CUDA runtime: /usr/local/cuda-13.0
```

Host has neither `torch` nor `transformers`; `uv` and `hf` are absent from PATH. This confirms
the standing rule that ML work here is **container-first**.

## Execution mode: containerized

Base image selected: **`nvcr.io/nvidia/pytorch:25.11-py3`** (19.5 GB, already resident on the
node). Chosen over the six local vLLM images because those are inference runtimes built around
vLLM, not training containers.

Images available on the node at the time of the probe:

```
vllm-node-b12x-nvfp4:latest        23.3GB
vllm-node-b12x-fix:latest          23.3GB
vllm-node-main:latest              23.3GB     <- currently serving gemma
eugr/spark-vllm-b12x:latest        23.3GB
vllm-node-b12x:latest              23.3GB
vllm-node-pr51959:latest           23.4GB
vllm/vllm-openai:v0.27.1-aarch64   22.5GB
nvcr.io/nvidia/pytorch:25.11-py3   19.5GB     <- selected
```

## Criterion 1 — transformers v5 resolves the model  ✅

```
transformers 5.15.0
accelerate 1.14.0

model_type: qwen3_5
arch: ['Qwen3_5ForConditionalGeneration']
```

`transformers` and `accelerate` are **not** in the NGC image; both installed cleanly from PyPI
on aarch64 with no build step and no wheel-availability problem.

`AutoConfig.from_pretrained("Qwen/Qwen3.8-27B")` resolved without `trust_remote_code`.

## Criterion 2 — GPU visible at sm_121  ✅

```
python 3.12.3
torch 2.10.0a0+b558c986e8.nv25.11
cuda_avail True
capability (12, 1)
device NVIDIA GB10
```

`(12, 1)` is sm_121, matching the required build target.

## Criterion 3 — architecture and size, measured from the checkpoint index

Top-level config is multimodal, so `num_hidden_layers` etc. are `None` at the root and live in
`text_config`. Reading the nested config:

```
num_hidden_layers:        64
hidden_size:              5120
intermediate_size:        17408
num_attention_heads:      24
num_key_value_heads:      4
head_dim:                 256
vocab_size:               248320
max_position_embeddings:  262144
full_attention_interval:  4
layer_types:              len=64  {'linear_attention': 48, 'full_attention': 16}
linear_num_key_heads:     16
linear_num_value_heads:   48
linear_key_head_dim:      128
linear_value_head_dim:    128
```

**The 48/16 layer split is now confirmed from the model's own config**, independently of the
companion inference repo's README. Q009 stands: a default LoRA preset targeting `q/k/v/o` +
`gate/up/down` reaches 16 of 64 layers.

From `model.safetensors.index.json` (a few KB — no weights downloaded):

```
total_size_bytes:  55562855904
total_size_GiB:    51.75
approx_params:     27.78 B  (bf16)
num_tensors:       1199
num_shards:        18

prefix model.language_model  850 tensors
prefix model.visual          333 tensors
prefix mtp.*                  13 tensors
prefix lm_head.weight          1 tensor
```

Module names present in the checkpoint, which is what LoRA target selection must actually key on:

```
 48  linear_attn.A_log        48  in_proj_a.weight       48  in_proj_qkv.weight
 48  conv1d.weight            48  in_proj_b.weight       48  in_proj_z.weight
 48  linear_attn.dt_bias      48  out_proj.weight
 17  q_norm.weight  17  k_norm.weight  17  k_proj.weight  17  o_proj.weight
 28  linear_fc1.weight  28  linear_fc2.weight  27  qkv.weight   (vision tower)
```

Note the recurrent layers expose `in_proj_a / in_proj_b / in_proj_qkv / in_proj_z / out_proj`,
**not** `q_proj/k_proj/v_proj`. A LoRA config written from muscle memory would silently miss all
48 of them.

## Findings that change downstream tasks

1. **BF16 base is 51.75 GiB, ~27.78 B params.** The 54 GB figure used until now was close but
   was arithmetic; this is read from the checkpoint index. Revised full-FT estimate:
   51.75 (weights) + 51.75 (grads, bf16) + ~51.75 (8-bit Adam, 2 states × 1 B/param)
   ≈ **155 GiB** before activations, gradient-checkpointing buffers and FSDP overhead.
   Still exceeds one node (121 GB ≈ 112.7 GiB) and still requires 2-node FSDP. **Q006 remains
   open** — this is a better estimate, not a measurement. P1.003 measures it.
2. **The checkpoint ships a vision tower (333 tensors) and an MTP head (13 tensors).** A
   text-only SFT should almost certainly freeze both. Freezing removes them from the gradient
   and optimizer-state budget, which is the cheapest available way to make the 2-node fit less
   marginal. This is a real design decision and belongs in P1.004, not in a default.
3. **`language_model_only` exists as a top-level config flag.** Worth testing in P1.003 — if it
   loads a text-only model, it may sidestep the vision tower entirely.

## Explicit non-results

- **No model weights were downloaded.** Only `config.json` and the safetensors index.
- **No training step has run.** Nothing here demonstrates that a 27B optimizer step fits; it
  demonstrates that the toolchain imports and the model resolves.
- **No memory profile was measured.** Every GiB figure above about *training* is arithmetic.
- The probe ran on **node 1 only**. Multi-node FSDP and the 200G fabric are untested here.
- `transformers` was installed at runtime into an ephemeral container; nothing is pinned yet.

## Required follow-up

**A run whose numbers get cited cannot depend on `pip install` at container start** — PyPI
resolves whatever is newest that day, so two runs a week apart would silently differ in
`transformers` version while reporting the same "environment". Build and tag a pinned training
image before P1.003. Added as TASK P1.007.

## Reproduce

```bash
sudo docker run --rm --gpus all --ipc=host nvcr.io/nvidia/pytorch:25.11-py3 bash -lc '
  pip install -q --no-cache-dir "transformers>=5.0" accelerate
  python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability())"
  python -c "from transformers import AutoConfig;print(AutoConfig.from_pretrained(\"Qwen/Qwen3.8-27B\").model_type)"
'
```
