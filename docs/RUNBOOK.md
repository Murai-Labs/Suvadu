# Suvadu — Runbook

Operational procedures. Exact, copy-pasteable commands only.

Sections marked **[UNVERIFIED]** have not yet been executed successfully on this hardware. They
are the intended procedure, not a record of one. Remove the marker only after the command has
run and its output has been pasted into the relevant gate review.

## Cluster access

Windows → Spark. Always `-T -n` (a forced TTY hangs). Windows mangles `|`, `{{}}` and `$()`, so
run any non-trivial remote script by base64-encoding locally and piping to `base64 -d | bash`.

```bash
ssh -T -n Murailabs-Spark    # node 1  spark-1003.local  192.168.4.26
ssh -T -n Murailabs-Spark2   # node 2  spark-e7ec.local  192.168.4.20
```

From PowerShell, for anything with pipes or braces:

```powershell
$s = @'
<remote script here>
'@
$b = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($s))
ssh -T -n Murailabs-Spark "echo $b | base64 -d | bash"
```

## Check whether the cluster is free

Run this before assuming anything about availability. It drifts.

```bash
ssh -T -n Murailabs-Spark  "free -g | head -2; ps -eo pid,rss,etime,comm --sort=-rss | head -5"
ssh -T -n Murailabs-Spark2 "free -g | head -2; ps -eo pid,rss,etime,comm --sort=-rss | head -5"
```

As of 2026-08-16 both nodes showed 111 GB of 121 GB used by a DeepSeek-V4-Flash TP=2 vLLM
deployment. Training cannot start until that is stopped (TASK P1.005).

## Restore the DeepSeek-V4-Flash deployment  **[UNVERIFIED]**

**This section must be completed with the exact working relaunch command BEFORE the deployment
is stopped.** Capturing it afterwards is not possible. Key known constraint: worker rank 1 must
be launched with `--headless`. Configuration lives at `~/deepseek-v4/` on the nodes; see the
`deepseek_v4_spark` operational notes.

```
TODO(P1.005): paste the verified relaunch command here before executing any shutdown.
Until this block is filled, stopping the deployment is forbidden.
```

## Environment setup  **[UNVERIFIED]**

ML on Spark is container-first — aarch64 + Blackwell wheels are unreliable, and builds must
match sm_121. Establish and pin the image in TASK P1.001 before filling this in.

```bash
# Determined by P1.001. Record the image digest or wheel set in configs/phase1/env-report.md.
```

Verification that must pass before the environment is considered usable:

```bash
python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('Qwen/Qwen3.8-27B').model_type)"
```

Expected: transformers v5+, capability `(12, 1)`, model_type `qwen3_5`.

## Download the trainable base model  **[UNVERIFIED]**

Every Qwen3.8-27B copy currently on any machine here is an inference quantization (NVFP4/GGUF)
and **cannot be fine-tuned**. The BF16 base is a separate ~54 GB download.

```bash
hf download Qwen/Qwen3.8-27B --local-dir ~/models/Qwen3.8-27B
du -sh ~/models/Qwen3.8-27B
```

Record the resolved revision SHA in `configs/phase1/base-model.json`. A pinned revision is
required — "latest" is not reproducible.

## Run a smoke test  **[UNVERIFIED]**

The smallest end-to-end command that exercises the pipeline. Always run this before a long job.

```bash
python -m suvadu.cli.train --config configs/phase1/smoke.yaml --max-steps 5
```

Must produce: a provenance manifest written *before* step 1, progress lines carrying
step/total + elapsed + ETA + loss, and a non-zero exit code on any missing identifier.

## Launch a full run  **[UNVERIFIED]**

```bash
python -m suvadu.cli.train --config configs/phase5/main.yaml --run-id phase5-main-001
```

## Resume an interrupted run  **[UNVERIFIED]**

```bash
python -m suvadu.cli.train --config configs/phase5/main.yaml --run-id phase5-main-001 --resume
```

Resume validates the config hash and seed against the recorded manifest and refuses to continue
on mismatch rather than silently starting a different experiment.

## Pre-Run Checklist

- [ ] `grep -r "SUVADU-PLACEHOLDER" src tests` is empty for reachable code.
- [ ] Config hash, code SHA, data hash, seed, environment recorded.
- [ ] Run ID allocated and never previously used.
- [ ] Progress logging emits every ≤100 steps.
- [ ] The gate governing this phase is approved in `docs/DECISION_LOG.md`.
- [ ] For any run past G5: G5 launch approval exists and the recipe hash is frozen.

## Reporting Checklist

Before any number reaches Ramchand, a README, or a model card:

- [ ] Re-read the artifact **now** — not the run output from when it was produced. Background
      jobs and other sessions overwrite files between production and reporting.
- [ ] Read at least three individual items before quoting any aggregate, rate, or tally.
- [ ] Timestamp anything that varies (process state, remote state, files another session touches).
- [ ] For comparisons: report the McNemar p and the discordant counts, never accuracy alone, and
      state the 6-discordant-item sensitivity floor.
