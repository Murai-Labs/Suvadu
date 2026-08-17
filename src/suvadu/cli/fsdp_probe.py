"""2-node FSDP memory probe (TASK P1.003, distributed stage).

Launched by torchrun, one process per node:

    torchrun --nnodes=2 --nproc-per-node=1 --node-rank=$RANK \\
             --master-addr=10.100.168.1 --master-port=29500 \\
             -m suvadu.cli.fsdp_probe --config configs/phase1/smoke.yaml \\
             --model-path /models/Qwen3.8-27B

Answers the half of Q006 the single-node probe could not: full-parameter training does not fit
on one node (measured 2026-08-17, OOM at 116.05 GiB of 121.69), so the question is whether
sharding across both nodes brings it under the ceiling.

Why this is a separate file from `memprobe.py`: that one works and is the record of the
single-node result. Distributed code fails in ways single-process code does not — rendezvous
hangs, one rank OOMing while the other waits forever — and mixing the two would put the working
measurement at risk of a regression in code it never runs.

Safety, learned the expensive way on 2026-08-17: an uncapped probe OOM on GB10 takes the *node*
down, not just the process, because the CUDA pool is system memory. Every rank sets a hard
allocator ceiling before touching the model.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

from suvadu.config import RunConfig
from suvadu.train.freeze import DEFAULT_POLICY, apply_freeze_policy

GIB = 1024 ** 3


def _mem() -> dict[str, float]:
    import torch

    out: dict[str, float] = {}
    if torch.cuda.is_available():
        out |= {
            "torch_allocated": round(torch.cuda.memory_allocated() / GIB, 2),
            "torch_reserved": round(torch.cuda.memory_reserved() / GIB, 2),
            "torch_max_allocated": round(torch.cuda.max_memory_allocated() / GIB, 2),
            "torch_max_reserved": round(torch.cuda.max_memory_reserved() / GIB, 2),
        }
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                if k in ("MemTotal", "MemAvailable"):
                    out[f"sys_{k}"] = round(int(rest.strip().split()[0]) / (1024 ** 2), 2)
    except OSError:
        pass
    return out


def _log(rank: int, msg: str) -> None:
    print(f"[fsdp_probe r{rank}] {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="suvadu-fsdp-probe", description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-class", choices=("causal_lm", "image_text_to_text"),
                   default="image_text_to_text",
                   help="Default keeps the vision tower (DEC 2026-08-17: keep vision). "
                        "causal_lm would silently make the published model text-only.")
    p.add_argument("--freeze-policy", default=DEFAULT_POLICY)
    p.add_argument("--seq-len", type=int, default=0)
    p.add_argument("--steps", type=int, default=3, help="optimizer steps to time")
    p.add_argument("--max-mem-fraction", type=float, default=0.85)
    p.add_argument("--out", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = RunConfig.from_yaml(args.config)
    seq_len = args.seq_len or cfg.max_seq_len

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)
    if 0 < args.max_mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.max_mem_fraction, local_rank)
    total = torch.cuda.get_device_properties(local_rank).total_memory / GIB
    _log(rank, f"world={world} device_total={total:.2f}GiB "
               f"ceiling={total * args.max_mem_fraction:.2f}GiB")

    t_rv = time.monotonic()
    dist.init_process_group(backend="nccl")
    _log(rank, f"rendezvous ok in {time.monotonic() - t_rv:.1f}s")

    stages: list[dict] = []

    def record(stage: str, t0: float, **detail) -> None:
        entry = {"stage": stage, "seconds": round(time.monotonic() - t0, 1),
                 "rank": rank, "mem": _mem(), **detail}
        stages.append(entry)
        _log(rank, f"{stage:10s} {entry['seconds']:7.1f}s "
                   f"alloc={entry['mem'].get('torch_allocated')}GiB "
                   f"sys_avail={entry['mem'].get('sys_MemAvailable')}GiB")

    # ---- load ---------------------------------------------------------------------------
    t0 = time.monotonic()
    if args.model_class == "causal_lm":
        from transformers import AutoModelForCausalLM as _Cls
    else:
        from transformers import AutoModelForImageTextToText as _Cls
    # Load STRAIGHT onto the device, never CPU-then-.to("cuda").
    #
    # This cost three OOMs on 2026-08-17 before the numbers gave it away. GB10 has unified
    # memory: there is no separate VRAM, so a CPU copy and a GPU copy occupy the *same* physical
    # pool. Loading to CPU and then moving to device holds the model TWICE — ~51 GiB of dead
    # weight for the lifetime of the run. The tell was torch accounting for 51.8 GiB while
    # /proc/meminfo reported only 15.95 GiB available: ~54 GiB in use that torch did not own,
    # almost exactly one model.
    #
    # On a discrete-GPU box this pattern is merely wasteful and transient. Here it is fatal, and
    # torch's own memory summary cannot see it — which is why this probe reads /proc/meminfo
    # alongside the allocator rather than trusting either alone.
    model = _Cls.from_pretrained(
        args.model_path,
        dtype=getattr(torch, cfg.dtype),
        device_map={"": f"cuda:{local_rank}"},
        low_cpu_mem_usage=True,
    )
    summary = apply_freeze_policy(model, args.freeze_policy)
    # TRAINING MODE FIRST. `from_pretrained` returns a model in eval mode, and HuggingFace's
    # checkpointed layers branch on `if self.gradient_checkpointing and self.training:`. With
    # the flag set but `training=False`, checkpointing is silently skipped and every activation
    # is retained — while `is_gradient_checkpointing` still reports True.
    #
    # That cost four OOMs on 2026-08-17. The read-back below confirms the FLAG; only
    # `model.training` confirms it will actually engage, so both are recorded.
    model.train()

    ckpt_on = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        ckpt_on = bool(getattr(model, "is_gradient_checkpointing", False))
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    record("load_cpu", t0, model_class=type(model).__name__,
           params_total=n_total, params_trainable=n_train,
           gradient_checkpointing_flag=ckpt_on,
           training_mode=bool(model.training),
           checkpointing_will_engage=bool(ckpt_on and model.training),
           trained_tensors=summary.counts_by_group, frozen_tensors=summary.frozen_by_group)
    if cfg.gradient_checkpointing and not (ckpt_on and model.training):
        _log(rank, "FATAL: gradient checkpointing requested but will NOT engage "
                   f"(flag={ckpt_on}, training={model.training}). Refusing to run — the "
                   "measurement would describe a different configuration than the one asked for.")
        raise SystemExit(3)

    # ---- shard --------------------------------------------------------------------------
    t0 = time.monotonic()
    try:
        # FSDP2 in torch 2.10. Falls back to FSDP1 if the newer API is absent, because a probe
        # that cannot run teaches nothing.
        from torch.distributed.fsdp import fully_shard
        for module in model.modules():
            if module.__class__.__name__.endswith(("DecoderLayer", "Block")):
                fully_shard(module)
        fully_shard(model)
        api = "fsdp2:fully_shard"
    except ImportError:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model = FSDP(model, device_id=local_rank, use_orig_params=True)
        api = "fsdp1:FullyShardedDataParallel"

    # No .to("cuda") here — the model is already on device (see the load comment). Reclaim
    # whatever the shard left behind before measuring, so the recorded figure is steady-state
    # rather than steady-state plus transient.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    record("shard", t0, api=api)

    # ---- optimizer ----------------------------------------------------------------------
    t0 = time.monotonic()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw_8bit":
        # torchao, NOT bitsandbytes.
        #
        # bitsandbytes' AdamW8bit cannot be used with FSDP2 (verified 2026-08-17):
        #   RuntimeError: bitsandbytes.optimizer_update_8bit_blockwise.default:
        #   got mixed torch.Tensor and DTensor
        # Its custom CUDA op does not dispatch over DTensor, and FSDP2 hands the optimizer
        # DTensor-sharded parameters. The Q010 check that "passed" used a plain Parameter, so it
        # confirmed the kernel runs — not that it runs on what FSDP2 actually produces.
        #
        # torchao's low-bit optimizers are DTensor-aware, and torchao ships inside the NGC base
        # image already, so this also removes the runtime pip install and makes the run
        # reproducible from the pinned image alone.
        from torchao.optim import AdamW8bit
        optimizer = AdamW8bit(trainable, lr=cfg.learning_rate)
        impl = "torchao.optim.AdamW8bit"
    else:
        # fp32 AdamW is recorded as unusable here, not merely unpreferred: two moments at
        # 4 bytes over ~13.4B sharded params is ~100 GiB of optimizer state per rank.
        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, fused=True)
        impl = "torch.optim.AdamW(fused)"
    record("optim_init", t0, optimizer=cfg.optimizer, impl=impl,
           n_trainable_tensors=len(trainable))

    # ---- steps --------------------------------------------------------------------------
    step_times: list[float] = []
    for i in range(1, args.steps + 1):
        t0 = time.monotonic()
        batch = torch.randint(1, 1000, (cfg.per_device_batch_size, seq_len),
                              device="cuda", dtype=torch.long)

        # Phase-by-phase on step 1. The 2026-08-17 run OOMed somewhere inside this block and the
        # traceback named only the line — which left forward, backward and the optimizer's lazy
        # state allocation indistinguishable. Splitting them means the next failure names the
        # term, which is the whole point of a staged probe.
        out = model(input_ids=batch, labels=batch)
        if i == 1:
            record("step1_forward", t0, note="after forward, before backward")
        out.loss.backward()
        if i == 1:
            record("step1_backward", t0, note="after backward, before optimizer.step")
        optimizer.step()
        if i == 1:
            record("step1_optim_step", t0,
                   note="after first optimizer.step; 8-bit state is allocated lazily HERE, "
                        "not at optim_init")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.monotonic() - t0
        step_times.append(dt)
        record(f"step{i}", t0, loss=float(out.loss.detach()),
               tokens=cfg.per_device_batch_size * seq_len * world,
               tokens_per_s=round(cfg.per_device_batch_size * seq_len * world / dt, 1))

    # Step 1 includes one-off allocation and autotuning; steady state is what a schedule is
    # built from, so report both rather than an average that hides the difference.
    steady = step_times[1:] or step_times
    summary_payload = {
        "rank": rank,
        "world_size": world,
        "model_class": args.model_class,
        "freeze_policy": args.freeze_policy,
        "seq_len": seq_len,
        "max_mem_fraction": args.max_mem_fraction,
        "device_total_gib": round(total, 2),
        "step_seconds_first": round(step_times[0], 2),
        "step_seconds_steady_mean": round(sum(steady) / len(steady), 2),
        "tokens_per_step_global": cfg.per_device_batch_size * seq_len * world,
        "peak_alloc_gib": max(s["mem"].get("torch_max_allocated", 0) for s in stages),
        "stages": stages,
    }
    _log(rank, "SUMMARY " + json.dumps({k: v for k, v in summary_payload.items()
                                        if k != "stages"}))

    if args.out and rank == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        _log(rank, f"wrote {args.out}")

    dist.barrier()
    dist.destroy_process_group()
    _log(rank, "exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
