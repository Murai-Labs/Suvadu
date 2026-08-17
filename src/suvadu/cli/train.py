"""Training entrypoint (TASK P1.004).

    python -m suvadu.cli.train --config configs/phase1/smoke.yaml --run-id phase1-smoke-001

Two modes:

``--plan``   Validate the config, resolve the freeze policy against the checkpoint's real
             parameter names, and print what *would* happen. Runs on any machine — no GPU, no
             weights, no torch. This is what you run before burning cluster time.

(default)    Train. Writes a provenance manifest **before** the first step and refuses to start
             without one, per AGENTS.md §2.4.

Order of operations matters and is enforced: config validated → resume checked → provenance
written → model loaded → freeze applied → first step. Loading a 51.75 GiB model and *then*
discovering the config is invalid wastes twenty minutes per mistake.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from suvadu.config import ConfigError, RunConfig
from suvadu.provenance import ProvenanceError, canonical_hash, write_manifest
from suvadu.train.data import BatchSpec, assert_not_synthetic_for_metrics, synthetic_batches
from suvadu.train.freeze import DEFAULT_POLICY, apply_freeze_policy, summarise_trainable
from suvadu.train.progress import ProgressReporter
from suvadu.train.resume import ResumeMismatch, ResumeState


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="suvadu-train", description=__doc__)
    p.add_argument("--config", required=True, help="path to the arm's YAML config")
    p.add_argument("--run-id", help="overrides the run_id in the config")
    p.add_argument("--runs-dir", default="runs", help="root directory for run artifacts")
    p.add_argument("--freeze-policy", default=DEFAULT_POLICY,
                   help="which sub-networks to train (see suvadu.train.freeze)")
    p.add_argument("--data", choices=("corpus", "synthetic"), default="corpus",
                   help="'synthetic' is for the P1.003 memory probe only")
    p.add_argument("--max-steps", type=int, default=0,
                   help="stop after N steps (0 = full run); used for smoke tests")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    p.add_argument("--plan", action="store_true",
                   help="validate and print the plan without training; no GPU required")
    p.add_argument("--param-names",
                   help="[--plan only] file of parameter names, one per line, to resolve the "
                        "freeze policy against real names rather than a guess")
    return p


def _load_config(path: str) -> RunConfig:
    try:
        return RunConfig.from_yaml(path)
    except ConfigError as exc:
        field = f" (field: {exc.fieldname})" if exc.fieldname else ""
        print(f"[suvadu] config rejected{field}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def cmd_plan(args: argparse.Namespace, cfg: RunConfig) -> int:
    """Print what a real run would do. No weights, no GPU, no side effects."""
    print(f"[suvadu] plan for arm={cfg.arm!r} run_id={cfg.run_id!r}")
    print(f"[suvadu]   base       {cfg.base_model}@{cfg.base_revision[:12]}")
    print(f"[suvadu]   regime     {cfg.regime}  dtype={cfg.dtype}  seq={cfg.max_seq_len}")
    print(f"[suvadu]   corpus     {cfg.corpus_path}  hash={cfg.corpus_hash[:12]}")
    print(f"[suvadu]   optimiser  {cfg.optimizer}  lr={cfg.learning_rate}  epochs={cfg.epochs}")
    print(f"[suvadu]   batching   per_device={cfg.per_device_batch_size} "
          f"grad_accum={cfg.gradient_accumulation_steps} ckpt={cfg.gradient_checkpointing}")
    print(f"[suvadu]   config_hash {canonical_hash(cfg.to_dict())}")

    if args.param_names:
        names = [ln.strip() for ln in Path(args.param_names).read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        summary = summarise_trainable(names, args.freeze_policy)
        print(f"[suvadu]   freeze     policy={summary.policy}")
        print(f"[suvadu]     trainable {summary.n_trainable} tensors  {summary.counts_by_group}")
        print(f"[suvadu]     frozen    {summary.n_frozen} tensors  {summary.frozen_by_group}")
    else:
        print(f"[suvadu]   freeze     policy={args.freeze_policy} "
              "(pass --param-names to resolve against real tensor names)")

    if args.data == "synthetic":
        print("[suvadu]   data       SYNTHETIC - memory profiling only; metrics will be refused")
    return 0


def cmd_train(args: argparse.Namespace, cfg: RunConfig) -> int:
    run_id = args.run_id or cfg.run_id
    run_dir = Path(args.runs_dir) / run_id
    config_hash = canonical_hash(cfg.to_dict())

    # --- resume check happens before anything expensive ------------------------------------
    start_step, start_epoch = 0, 0
    if args.resume:
        state = ResumeState.read(run_dir)
        if state is None:
            print(f"[suvadu] --resume given but no resume state in {run_dir}", file=sys.stderr)
            return 2
        try:
            state.validate_against(
                run_id=run_id, config_hash=config_hash,
                data_hash=cfg.corpus_hash, seed=cfg.seed,
            )
        except ResumeMismatch as exc:
            print(f"[suvadu] {exc}", file=sys.stderr)
            return 2
        start_step, start_epoch = state.global_step, state.epoch
        print(f"[suvadu] resuming at epoch {start_epoch} step {start_step}")

    # --- provenance is a precondition, not a postscript -------------------------------------
    if not args.resume:
        try:
            manifest_path = write_manifest(
                run_dir,
                run_id=run_id,
                config=cfg.to_dict(),
                data_hash=cfg.corpus_hash,
                seed=cfg.seed,
                notes={"freeze_policy": args.freeze_policy, "data_source": args.data},
            )
        except ProvenanceError as exc:
            print(f"[suvadu] refusing to start: {exc}", file=sys.stderr)
            return 2
        print(f"[suvadu] provenance written: {manifest_path}")

    # --- from here on we need torch ----------------------------------------------------------
    import torch
    from transformers import AutoModelForCausalLM

    print(f"[suvadu] loading {cfg.base_model}@{cfg.base_revision[:12]} dtype={cfg.dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        revision=cfg.base_revision,
        dtype=getattr(torch, cfg.dtype),
    )

    summary = apply_freeze_policy(model, args.freeze_policy)
    print(f"[suvadu] freeze policy={summary.policy} "
          f"trainable={summary.n_trainable} frozen={summary.n_frozen}")
    (run_dir / "freeze_summary.json").write_text(
        json.dumps(
            {
                "policy": summary.policy,
                "trained_groups": list(summary.trained_groups),
                "counts_by_group": summary.counts_by_group,
                "frozen_by_group": summary.frozen_by_group,
                "n_trainable_tensors": summary.n_trainable,
                "n_frozen_tensors": summary.n_frozen,
                "n_trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "n_total_params": sum(p.numel() for p in model.parameters()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.data == "synthetic":
        if args.max_steps <= 0:
            print("[suvadu] --data synthetic requires --max-steps; it is a probe, not a run",
                  file=sys.stderr)
            return 2
        spec = BatchSpec(
            batch_size=cfg.per_device_batch_size,
            seq_len=cfg.max_seq_len,
            vocab_size=248320,
        )
        batches = synthetic_batches(spec, seed=cfg.seed, n_batches=args.max_steps)
        total_steps = args.max_steps
    else:
        from suvadu.train.data import corpus_batches
        batches = corpus_batches(cfg)      # raises until G2; see AGENTS.md 2.3
        total_steps = args.max_steps or 0

    reporter = ProgressReporter(total_steps=total_steps, log_every=cfg.log_every_n_steps)
    reporter.start()

    print(f"[suvadu] beginning {total_steps} steps from step {start_step}")
    for i, batch in enumerate(batches, start=start_step + 1):
        # The optimizer/FSDP wiring lands with P1.003, which measures what actually fits before
        # the schedule is fixed. Until then this loop exercises data, freeze and reporting.
        step_loss = float("nan")
        reporter.log(
            i,
            loss=None if step_loss != step_loss else step_loss,
            peak_mem_gib=(torch.cuda.max_memory_allocated() / 1024**3
                          if torch.cuda.is_available() else None),
        )
        if args.max_steps and i >= start_step + args.max_steps:
            break

    assert_not_synthetic_for_metrics(args.data, purpose="training metrics")
    ResumeState.create(
        run_id=run_id, config_hash=config_hash, data_hash=cfg.corpus_hash,
        seed=cfg.seed, epoch=start_epoch, global_step=start_step + total_steps,
    ).write(run_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(args.config)
    return cmd_plan(args, cfg) if args.plan else cmd_train(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
