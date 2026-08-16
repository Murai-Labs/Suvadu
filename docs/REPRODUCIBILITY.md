# Suvadu — Reproducibility

## Required Run Evidence

Every run records: git commit, exact command, config path + config hash, data split id + data
hash, seed, environment (GPU, CUDA, Python, package versions), checkpoint path, metrics path,
failure notes (if any), interpretation, next-step recommendation.

## Artifact Integrity

- Configs are copied into the run dir or immutably referenced; immutable after launch.
- Train/eval data splits are physically separate; no overlap. For this project that is enforced
  by the decontamination pass (TASK P2.006), not assumed.
- Metrics, figures, tables, and reports cite the run IDs that produced them.
- Any result file lacking the 5 provenance identifiers → `notes/untrusted-results.md`, excluded
  from the report and the model card.

## Deterministic Rerun Checklist

- [ ] Same git commit
- [ ] Same config hash
- [ ] Same data hash
- [ ] Same seed
- [ ] Same command
- [ ] Same dependency versions
- [ ] Same hardware class

## Project-Specific Reproducibility Hazards

These are known ways a Suvadu result could look reproducible and not be:

1. **The corpus is regenerated from a live machine.** `build_traces_dataset.py` reads session
   traces that keep accumulating. Two exports a week apart are different datasets. Every arm must
   cite a corpus hash, and the hash is frozen at G2. Never re-export mid-experiment.
2. **The base checkpoint must be revision-pinned.** `Qwen/Qwen3.8-27B` without a revision SHA is
   not a reproducible input.
3. **Quantized checkpoints are not the model under test.** The NVFP4 and GGUF copies on these
   machines are inference artifacts. A result measured on NVFP4 and a result measured on BF16 are
   not comparable, and mixing them silently would invalidate a comparison.
4. **Chat-template rendering is part of the data.** If training-time rendering and inference-time
   rendering differ (see Q007), the same weights produce different scores. The rendering used
   must be recorded alongside the corpus hash.
5. **The cluster is shared.** Runs can be affected by whatever else is resident. Record what else
   was running, not just that the run finished.

## Reporting Gate

No metric is citable unless it has: run ID, metric file path, config hash, code SHA, data hash,
seed, environment record, and stated known limitations.

For any **comparative** claim — the entire point of this project — additionally required:
per-item results for both arms (so the comparison can be paired), the McNemar p-value, the
discordant-item counts, and the stated sensitivity floor of 6 discordant items.

An accuracy difference reported without its discordant counts is not a result. Two independent
runs at n=200 can differ by 4 points with fully overlapping confidence intervals and mean
nothing; the items that *flip* are the signal.
