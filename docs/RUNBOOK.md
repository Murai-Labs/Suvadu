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

As of 2026-08-16 both nodes showed 111 GB of 121 GB used. Training cannot start until the
resident workload is stopped (TASK P1.005).

## Restore the Gemma deployment currently occupying the cluster

**Verified by `docker inspect` on 2026-08-16, before any shutdown.**

What is actually running is **not** a DeepSeek-V4-Flash TP=2 deployment. It is **two
independent single-node vLLM servers**, one per node, each serving `google/gemma-4-31B-it` on
its own `:8000`. There is no `--tensor-parallel-size`; the nodes are not coupled. Either can be
stopped without affecting the other.

Container: `vllm-gemma` · image `vllm-node-main:latest` · started 2026-08-13T21:29Z ·
**`restart=no`** — once stopped it stays down until relaunched by hand.

Relaunch, run separately on **each** node:

```bash
sudo docker run -d --name vllm-gemma \
  --gpus all --ipc=host -p 8000:8000 \
  -v /home/murailabs/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  vllm-node-main:latest \
  vllm serve google/gemma-4-31B-it --served-model-name gemma-4-31B-it \
    --gpu-memory-utilization 0.85 --max-model-len 8192 --max-num-seqs 64
```

Stop (per node):

```bash
sudo docker stop vllm-gemma && sudo docker rm vllm-gemma
```

Confirm it is back:

```bash
curl -s http://127.0.0.1:8000/v1/models
# expect: {"object":"list","data":[{"id":"gemma-4-31B-it", ...}]}
```

Two notes carried from the original (wrong) version of this section, kept because they apply to
a *different* deployment that also lives on this cluster: the DeepSeek-V4-Flash stack at
`~/deepseek-v4/` serves on **port 8888**, requires `--headless` on worker rank 1, and must be
torn down on both nodes. That stack is **not currently running** and is not what P1.005 stops.

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

## Download the trainable base model

Every Qwen3.8-27B copy on any machine here is an inference quantization (NVFP4/GGUF) and
**cannot be fine-tuned**. The BF16 base is a separate **51.75 GiB** download.

**`hf` is not on PATH on the Spark nodes** and the host has no `huggingface_hub`, so run the
download inside a container that already has it. Node 1 has `nvcr.io/nvidia/pytorch:25.11-py3`;
node 2 does not, but has `vllm-node-main:latest`, which also carries `huggingface_hub`. Use
whichever is already present — pulling a 19.5 GB image just to fetch weights is wasted bandwidth.

Script (written to `~/suvadu_download.py` on the node):

```python
import os, time
from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen3.8-27B"
REV  = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"   # pinned; repo was modified 2026-08-14
DEST = "/models/Qwen3.8-27B"

t0 = time.time()
print("[suvadu] start %s@%s -> %s" % (REPO, REV[:12], DEST), flush=True)
p = snapshot_download(repo_id=REPO, revision=REV, local_dir=DEST, max_workers=8)
el = time.time() - t0
total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
print("[suvadu] DONE in %.1f min" % (el / 60.0), flush=True)
print("[suvadu] bytes=%d (%.2f GiB)" % (total, total / 1024.0**3), flush=True)
```

Launch detached, per node:

```bash
mkdir -p $HOME/models
sudo docker run -d --name suvadu-dl1 \
  -e HF_HUB_OFFLINE=0 \
  -v $HOME/models:/models \
  -v $HOME/suvadu_download.py:/suvadu_download.py:ro \
  nvcr.io/nvidia/pytorch:25.11-py3 python /suvadu_download.py    # node 1
  # node 2: swap image for vllm-node-main:latest and `python` for `python3`
```

Watch it:

```bash
sudo docker logs --tail 5 suvadu-dl1
du -sh $HOME/models/Qwen3.8-27B
```

**Run it detached (`-d`), not in the SSH foreground.** A foreground `docker run` dies with the
SSH session — this is how the first node-2 attempt lost a 19.5 GB image pull partway through.

The revision SHA is pinned deliberately: the upstream repo was last modified **2026-08-14**, and
`main` moved **six times in a single day**. Record the resolved SHA and the on-disk byte count in
`configs/phase1/base-model.json`.

Measured 2026-08-17: **84 minutes per node**, 55,586,114,863 bytes, 32 files, identical totals on
both nodes.

### Verifying the download — and the CRC32 trap

`Qwen/Qwen3.8-27B` ships a `crc32.txt`. **Three of its eight entries fail, and that is expected.**

```
FAIL chat_template.jinja        FAIL generation_config.json     FAIL tokenizer_config.json
```

Do **not** re-download. The manifest is stale upstream: `crc32.txt` arrived in commit
`72a217afab80` (2026-08-13 08:23), and those three files were each replaced ~2 hours later in
`452438ec4174`, `412f8b6bd7f5` and `dbdc473dea0d`, without the manifest being regenerated. The
failing set is exactly the set of files modified after it was written.

Note also that **`crc32.txt` does not cover the weight shards at all** — it lists 8 config and
tokenizer files. It cannot tell you whether 51.75 GiB of weights arrived intact.

What actually gives confidence, and what this project uses instead:

```bash
# 1. byte totals identical across two independent downloads
find $HOME/models/Qwen3.8-27B -type f -not -path "*/.cache/*" -printf "%s\n" \
  | awk '{s+=$1} END {printf "bytes=%d files=%d\n", s, NR}'
# expect: bytes=55586114863 files=32

# 2. every shard the index references is present, and the tensor count matches
python -c "
import json,os; D='$HOME/models/Qwen3.8-27B'
i=json.load(open(D+'/model.safetensors.index.json')); f=set(i['weight_map'].values())
print('shards',len(f),'missing',sum(not os.path.exists(D+'/'+x) for x in f),'tensors',len(i['weight_map']))"
# expect: shards 18 missing 0 tensors 1199

# 3. config + tokenizer instantiate from the local dir
python -c "
from transformers import AutoConfig, AutoTokenizer; D='$HOME/models/Qwen3.8-27B'
print(AutoConfig.from_pretrained(D).model_type)
t=AutoTokenizer.from_pretrained(D); print(type(t).__name__, t.vocab_size)"
# expect: qwen3_5 / Qwen2Tokenizer 248044
```

`tokenizer.vocab_size` (248044) is smaller than the config's `vocab_size` (248320). That is the
ordinary embedding-padding gap, not a mismatch — the config sizes the padded embedding matrix,
the tokenizer reports the base vocabulary.

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
