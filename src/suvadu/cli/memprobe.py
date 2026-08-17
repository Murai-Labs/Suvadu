"""Memory probe (TASK P1.003) — does a 27B optimizer step actually fit?

    python -m suvadu.cli.memprobe --stage weights --config configs/phase1/smoke.yaml

This is the measurement that Q006 has been waiting on. Everything said about 155 GiB so far is
arithmetic; this replaces it with numbers.

Staged deliberately. A single "does it fit" run that OOMs tells you almost nothing — you learn
that some unidentified total exceeded some limit. Running weights → grads → optim → step in
order means an OOM identifies *which term* broke the budget, and every stage before it is
recorded regardless.

    weights   load the model, measure resident bytes
    grads     + one backward pass, measure the gradient term
    optim     + optimizer state materialised (8-bit Adam by default)
    step      + a full optimizer step, end to end

GB10 has unified CPU/GPU memory, so `torch.cuda` accounting alone is misleading here. Every
stage records both the torch allocator's view and the system's, and they are reported
separately rather than reconciled — if they disagree that is a finding, not an error.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from suvadu.config import RunConfig
from suvadu.train.freeze import DEFAULT_POLICY, apply_freeze_policy

GIB = 1024 ** 3
STAGES = ("weights", "grads", "optim", "step")


def _sys_mem_gib() -> dict[str, float]:
    """System memory from /proc/meminfo, in GiB.

    Read directly rather than via psutil: this must work in a container with no extra
    dependencies, and on unified memory the system view is the one that actually bounds us.
    """
    info: dict[str, float] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemFree", "MemAvailable"):
                    info[key] = int(rest.strip().split()[0]) / (1024 ** 2)
    except OSError:
        return {}
    return info


def _torch_mem_gib() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "allocated": torch.cuda.memory_allocated() / GIB,
        "reserved": torch.cuda.memory_reserved() / GIB,
        "max_allocated": torch.cuda.max_memory_allocated() / GIB,
        "max_reserved": torch.cuda.max_memory_reserved() / GIB,
    }


@dataclass
class StageResult:
    stage: str
    ok: bool
    seconds: float
    torch_mem_gib: dict[str, float] = field(default_factory=dict)
    sys_mem_gib: dict[str, float] = field(default_factory=dict)
    detail: dict = field(default_factory=dict)
    error: str | None = None


def _record(stage: str, t0: float, ok: bool, **detail) -> StageResult:
    r = StageResult(
        stage=stage,
        ok=ok,
        seconds=round(time.monotonic() - t0, 1),
        torch_mem_gib={k: round(v, 2) for k, v in _torch_mem_gib().items()},
        sys_mem_gib={k: round(v, 2) for k, v in _sys_mem_gib().items()},
        detail=detail,
    )
    print(f"[memprobe] {stage:8s} ok={ok} {r.seconds:7.1f}s "
          f"torch_alloc={r.torch_mem_gib.get('allocated', float('nan')):.2f}GiB "
          f"sys_avail={r.sys_mem_gib.get('MemAvailable', float('nan')):.2f}GiB",
          flush=True)
    return r


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="suvadu-memprobe", description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--model-path", required=True, help="local path to the BF16 checkpoint")
    p.add_argument("--stage", choices=STAGES, default="step",
                   help="highest stage to attempt; earlier stages always run")
    p.add_argument("--freeze-policy", default=DEFAULT_POLICY)
    p.add_argument("--model-class", choices=("causal_lm", "image_text_to_text"),
                   default="causal_lm",
                   help="causal_lm loads the text tower alone (26.90B, no vision, no MTP); "
                        "image_text_to_text keeps the vision tower (27.36B). Neither loads the "
                        "MTP head. Measured on a meta device 2026-08-17.")
    p.add_argument("--seq-len", type=int, default=0, help="override config max_seq_len")
    p.add_argument("--out", default="", help="write results JSON here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = RunConfig.from_yaml(args.config)
    seq_len = args.seq_len or cfg.max_seq_len
    target = STAGES.index(args.stage)
    results: list[StageResult] = []

    import torch

    print(f"[memprobe] torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"cap={torch.cuda.get_device_capability() if torch.cuda.is_available() else None}",
          flush=True)
    print(f"[memprobe] baseline sys mem: {_sys_mem_gib()}", flush=True)
    print(f"[memprobe] plan: stages up to {args.stage!r}, seq_len={seq_len}, "
          f"freeze={args.freeze_policy}", flush=True)

    model = optimizer = None

    # ---- stage: weights -----------------------------------------------------------------
    t0 = time.monotonic()
    try:
        if args.model_class == "causal_lm":
            from transformers import AutoModelForCausalLM as _Cls
        else:
            from transformers import AutoModelForImageTextToText as _Cls
        model = _Cls.from_pretrained(
            args.model_path, dtype=getattr(torch, cfg.dtype), device_map=None,
        ).to("cuda")
        summary = apply_freeze_policy(model, args.freeze_policy)

        # Apply what the config asks for. Measuring with checkpointing off while the config says
        # on would produce an activation figure that describes a run nobody intends to launch.
        ckpt_enabled = False
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False   # incompatible with checkpointing; warns and slows
            ckpt_enabled = getattr(model, "is_gradient_checkpointing", True)

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        results.append(_record(
            "weights", t0, True,
            model_class=type(model).__name__,
            params_total=n_total, params_trainable=n_train,
            params_frozen=n_total - n_train,
            trainable_fraction=round(n_train / n_total, 4),
            freeze_policy=summary.policy,
            trained_tensors=summary.counts_by_group,
            frozen_tensors=summary.frozen_by_group,
            gradient_checkpointing=ckpt_enabled,
        ))
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        results.append(_record("weights", t0, False))
        results[-1].error = f"{type(exc).__name__}: {exc}"
        print(f"[memprobe] weights FAILED: {results[-1].error}", flush=True)
        return _finish(args, results, 1)

    if target == 0:
        return _finish(args, results, 0)

    # ---- stage: grads -------------------------------------------------------------------
    t0 = time.monotonic()
    try:
        batch = torch.randint(1, 1000, (1, seq_len), device="cuda", dtype=torch.long)
        out = model(input_ids=batch, labels=batch)
        out.loss.backward()
        grad_bytes = sum(p.grad.numel() * p.grad.element_size()
                         for p in model.parameters() if p.grad is not None)
        results.append(_record("grads", t0, True,
                               loss=float(out.loss.detach()),
                               grad_bytes=grad_bytes,
                               grad_gib=round(grad_bytes / GIB, 2)))
    except Exception as exc:  # noqa: BLE001
        results.append(_record("grads", t0, False))
        results[-1].error = f"{type(exc).__name__}: {exc}"
        print(f"[memprobe] grads FAILED: {results[-1].error}", flush=True)
        return _finish(args, results, 1)

    if target == 1:
        return _finish(args, results, 0)

    # ---- stage: optim -------------------------------------------------------------------
    t0 = time.monotonic()
    try:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if cfg.optimizer == "adamw_8bit":
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable, lr=cfg.learning_rate)
        else:
            optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, fused=True)
        results.append(_record("optim", t0, True, optimizer=cfg.optimizer,
                               n_trainable_tensors=len(trainable)))
    except Exception as exc:  # noqa: BLE001
        results.append(_record("optim", t0, False))
        results[-1].error = f"{type(exc).__name__}: {exc}"
        print(f"[memprobe] optim FAILED: {results[-1].error}", flush=True)
        return _finish(args, results, 1)

    if target == 2:
        return _finish(args, results, 0)

    # ---- stage: step --------------------------------------------------------------------
    t0 = time.monotonic()
    try:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        results.append(_record("step", t0, True))
    except Exception as exc:  # noqa: BLE001
        results.append(_record("step", t0, False))
        results[-1].error = f"{type(exc).__name__}: {exc}"
        print(f"[memprobe] step FAILED: {results[-1].error}", flush=True)
        return _finish(args, results, 1)

    return _finish(args, results, 0)


def _finish(args, results: list[StageResult], code: int) -> int:
    payload = {
        "stages": [asdict(r) for r in results],
        "highest_stage_ok": next((r.stage for r in reversed(results) if r.ok), None),
        "exit_code": code,
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[memprobe] wrote {args.out}", flush=True)
    print("[memprobe] RESULT " + json.dumps(payload["stages"][-1] if results else {}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
