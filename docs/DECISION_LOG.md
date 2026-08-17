# Suvadu — Decision Log

## Update Rules

Record every material research, tooling, gate, execution, blocker-resolution, or reporting
decision here unless it already has a dedicated file under `docs/decisions/` (which must be
linked from this log). Record the rationale **before** running expensive jobs. Never overwrite
an entry — correct a past entry by appending a new one that references it.

Each entry includes: date, task/gate, decision, rationale, alternatives considered, evidence,
measured result (if any), follow-up, and human-approval status.

---

## DEC-0001 — Project scaffolding established

Date: 2026-08-16
Task/Gate: G0
Decision: Initialized the Suvadu repo with the standard Murai Labs governance/tracker/docs/notes
scaffolding and the phase→gate chain G0–G7.

Rationale: The project is a research experiment whose deliverable is a *public comparative
claim* ("beats base Qwen3.8-27B on tool calling and coding"). Public claims require an auditable
record: pre-registered thresholds, provenanced runs, and gate decisions taken before results are
seen. The standing lab scaffolding provides that.

Alternatives Considered:
- Plain repo with a README — rejected; a published comparative claim with no gate record is
  exactly the failure mode the lab's cheap-baseline rule exists to prevent.
- Reusing `Murai-Labs/qwen3.8-27b-rtx5090` — rejected; that repo is an *inference* benchmarking
  project with its own leaderboard, and mixing a training project into it would muddle both. Its
  eval harness is reused as a dependency instead (TASK P3.002).

Evidence / Source Docs: `notes/spec-comprehension-check.md`, `tasks/atomic-task-list.md`,
`docs/GATE_G0_REVIEW.md`.
Measured Result: N/A (setup).
Follow-up: Close G0 (P0.005), then G1 training-stack verification.
Human Approval: Approved by Ramchand on 2026-08-16 (scaffolding and public-repo decision).

---

## DEC-0002 — Gate chain deviates from the template: cheap-baseline gate is G4, not G1

Date: 2026-08-16
Task/Gate: G0
Decision: The mandatory cheap-baseline-falsification gate is numbered **G4** in this project,
not G1 as in the standard template.

Rationale: On this project the cheap baselines are themselves training runs (B1 token-matched
control, B3 subsample, B4 LoRA). They physically cannot execute before a verified training stack
(G1), a frozen corpus (G2), and a frozen evaluation with a measured base reference (G3) exist.
Numbering the gate G1 while placing it fourth in dependency order would misrepresent the chain.

Alternatives Considered:
- Keep the cheap-baseline gate at G1 and renumber the prerequisites — rejected; it would make
  the gate numbers non-monotonic with execution order, which is worse for a resuming agent.
- Drop the prerequisites and run baselines first — impossible; there is nothing to run them on.

Evidence / Source Docs: `CLAUDE.md` §3 and §7; `tasks/atomic-task-list.md` Phase 4.
Measured Result: N/A.
Follow-up: The deviation is stated in `CLAUDE.md` §3 and the README so it cannot read as the
gate being quietly demoted.
Human Approval: Approved by Ramchand on 2026-08-16.

---

## DEC-0003 — ε pinned at 3.0 percentage points, derived from a measured sensitivity floor

Date: 2026-08-16
Task/Gate: G4 (pre-registered at G0)
Decision: The equivalence margin for the G4 falsification decision is **ε = 3.0 percentage
points on metric M at n = 200**.

Rationale: `Murai-Labs/qwen3.8-27b-rtx5090`'s `bench/LEADERBOARD.md` documents a *measured*
sensitivity floor for its paired McNemar harness: 6 discordant items reach p<0.05 and 5 do not,
independent of n — verified by degrading a stored run item by item. At n = 200, six items is
exactly 3.0 pp. A difference smaller than that is invisible to the test, so treating it as real
would be unsupportable.

Alternatives Considered:
- A conventional ε of 1 pp — rejected; below the harness's demonstrated resolution at n=200, so
  the gate decision would be unadjudicable.
- Defer ε until results are in — rejected outright; a threshold set after seeing results is not
  a threshold. This is the specific rationalization the lab rule exists to block.

Evidence / Source Docs: `Murai-Labs/qwen3.8-27b-rtx5090` `bench/LEADERBOARD.md`, read 2026-08-16.
Measured Result: N/A (pre-registration).
Follow-up: TASK P3.004 computes the n required to resolve smaller differences; if a smaller ε is
wanted, n must rise first and the change must be recorded before Phase 4 runs.
Human Approval: Approved by Ramchand on 2026-08-16.

---

## DEC-0004 — Public repo, but internal trackers and audit trail stay local

Date: 2026-08-16
Task/Gate: G0
Decision: `Murai-Labs/Suvadu` is public. `STATUS.md`, `CHECKPOINT.md`, `GAPS.md` and the whole
of `notes/` are gitignored. `docs/` **is** committed.

Rationale: The standing Murai Labs rule keeps internal tracking files out of public repos. But
the decision log and gate reviews are precisely what make a public comparative claim auditable —
suppressing them would gut the reason for publishing at all. The split is therefore: internal
working state stays local; the research record is public.

Alternatives Considered:
- Private repo — rejected by Ramchand; the project is intended to be public.
- Commit everything including `notes/` — rejected; `notes/stuck-log.md`,
  `notes/integrity-gaps.md` and `notes/untrusted-results.md` are internal working artifacts that
  reference private paths and unpublished projects.

Evidence / Source Docs: `.gitignore`; global CLAUDE.md "Working style"; memory
`feedback_internal_tracking_files.md`.
Measured Result: N/A.
Follow-up: TASK P0.006 verifies with `git status --porcelain --ignored` before the first push.
Human Approval: Approved by Ramchand on 2026-08-16.

---

## DEC-0005 — DeepSeek-V4-Flash shutdown approved, execution deferred

Date: 2026-08-16
Task/Gate: G1
Decision: Ramchand authorized stopping the DeepSeek-V4-Flash TP=2 vLLM deployment occupying both
Spark nodes. Execution is **deferred** to TASK P1.005, after the BF16 weights are downloaded and
a training entrypoint exists.

Rationale: Verified on 2026-08-16: both `spark-1003` and `spark-e7ec` report 111 GB of 121 GB in
use, with `VLLM::EngineCore` resident for 2d21h and `:8000` listening on node 1. Training a 27B
full-parameter model needs that memory. But the setup work ahead — a ~54 GB download and toolchain
validation on aarch64 — does not need the GPUs. Stopping the endpoint now would cost days of
availability for no gain.

Alternatives Considered:
- Stop immediately on approval — rejected; wastes the endpoint during setup.
- Train on one node while DeepSeek runs on the other — rejected; the DeepSeek deployment is TP=2
  and spans both nodes, so it cannot survive on one.

Evidence / Source Docs: `ps`/`free`/`ss` output from both nodes, 2026-08-16; `STATUS.md`
"Verified Environment Facts".
Measured Result: N/A.
Follow-up: P1.005 must capture the exact relaunch command — including `--headless` on worker
rank 1 — *before* shutdown.
Human Approval: Approved by Ramchand on 2026-08-16.

---

## DEC-0006 — Correction to DEC-0005: the cluster is running Gemma, not DeepSeek

Date: 2026-08-16
Task/Gate: G1
Decision: DEC-0005 described the workload occupying both Spark nodes as a "DeepSeek-V4-Flash
TP=2 vLLM deployment". That is **wrong**. This entry corrects it; per the append-only rule,
DEC-0005 is left standing rather than edited.

What is actually running, verified by `curl /v1/models` and `docker inspect` on 2026-08-16:

```
vllm serve google/gemma-4-31B-it --served-model-name gemma-4-31B-it \
  --gpu-memory-utilization 0.85 --max-model-len 8192 --max-num-seqs 64
```

Container `vllm-gemma`, image `vllm-node-main:latest`, started 2026-08-13T21:29Z, `restart=no`.
**Two independent single-node servers**, one per node, each bound to its own `:8000`. No
tensor-parallel flag; the nodes are not coupled.

Rationale for recording this as a decision entry rather than a silent fix: the error changed the
options presented to Ramchand. He was told that keeping the endpoint alive on one node was
impossible because "TP=2 needs both nodes". That was false — the servers are independent and
either could have been kept up. He chose to free the whole cluster anyway, and full-parameter FT
of a 27B does need both nodes, so the decision itself stands. But it was made on a wrong premise
and the record must say so.

How the error happened: the claim was inferred from the HF cache contents on the nodes
(`models--deepseek-ai--DeepSeek-V4-Flash-0731` is present) plus a stored memory describing a
DeepSeek TP=2 recipe, and then stated as fact without querying the endpoint. Reading
`ps` output showing `VLLM::EngineCore` confirmed *that vLLM was running*, not *what it served*.
One `curl` would have settled it and was not run until later.

Alternatives Considered: none — this is a factual correction, not a choice.
Evidence / Source Docs: `curl http://127.0.0.1:8000/v1/models` on both nodes;
`docker inspect vllm-gemma`; `docs/RUNBOOK.md` restore section (now verified, not [UNVERIFIED]).
Measured Result: N/A.
Follow-up: `docs/RUNBOOK.md` rewritten with the verified relaunch command, which satisfies
P1.005's precondition (capture the restore procedure *before* shutdown) ahead of schedule.
TASK P1.005 retitled accordingly.
Human Approval: N/A — correction of fact, reported to Ramchand 2026-08-16.

---

## DEC-0007 — G0 approved

Date: 2026-08-16
Task/Gate: G0
Decision: G0 (Repository Skeleton) approved; Phase 1 training-stack bring-up unblocked.
Rationale: All six Phase 0 tasks complete with evidence tabulated in `docs/GATE_G0_REVIEW.md`
from artifacts re-read at the time of writing. 39 tests passing (19 config / 14 provenance /
6 import); `CLAUDE.md` and `AGENTS.md` byte-identical by SHA256; 31 files tracked locally and 31
on the remote, with `git check-ignore` confirming trackers and `notes/` are excluded.
Alternatives Considered: N/A.
Evidence / Source Docs: `docs/GATE_G0_REVIEW.md`, `tasks/atomic-task-list.md`, commit `a096d4d`.
Measured Result: N/A (setup gate).
Follow-up: P1.001.
Human Approval: **Approved by Ramchand on 2026-08-16.**

---

## DEC-0008 — r0b0tlab datasets approved for use despite absent licences

Date: 2026-08-16
Task/Gate: G2 (bearing on G7)
Decision: The seven `r0b0tlab/*` datasets are **approved for inclusion** in the training
mixture, reversing the default-EXCLUDED posture recorded when Q005 was opened.

Rationale: Ramchand approved them on 2026-08-16, stating that the dataset author is a friend who
omitted the licence file rather than withheld it, and that he approves proceeding on that basis.
That is the maintainer's call to make and it is recorded as made.

Stated precisely, because G7 requires redistribution status **in writing** and this is not that:
the approval is a *personal assurance relayed by the project owner*, not a licence grant. As of
2026-08-16 the dataset pages display no licence. Nothing about training is blocked by this. What
is affected is publication: a model card asserting its training data is redistributable would,
today, be asserting something for which no artifact exists.

Alternatives Considered:
- Keep them excluded — rejected by Ramchand; the datasets are wanted and the omission is known
  to be clerical.
- Include silently and note nothing — rejected; G7 would then be adjudicated against a record
  that does not mention the gap.

Evidence / Source Docs: HF dataset pages fetched 2026-08-16, all showing no licence
(`deepseek-v4-pro-0813-agentic`, `qwen3.8-max-glm5.2-kimi-k3-distillation`,
`qwen3.8-max-distillation-50k`, `deepseek-hermes-reasoning-traces`,
`nemotron-nano-hermes-traces`, `Hermes-OmniForge-Qwen36-27B-full-v0.3.0-unsloth`,
`gemma-4-e4b-hermes-agent-traces-reformatted`). Q005 in `docs/RISKS_AND_OPEN_QUESTIONS.md`.
Measured Result: N/A.
Follow-up: **P2.003 is narrowed, not closed.** Its remaining job before G7 is to obtain one
written confirmation from the author — a licence file added to the repos, or a message granting
redistribution — and to record which. Training proceeds now regardless. Also unchanged: these
are distillations of other vendors' model outputs, which is a separate ToS question from
licensing and is not resolved by the author's permission.
Human Approval: **Approved by Ramchand on 2026-08-16.**

---

## DEC-0009 — Gemma endpoints stopped to free the cluster for training

Date: 2026-08-17
Task/Gate: G1 (P1.005)
Decision: Stop both `vllm-gemma` containers (`google/gemma-4-31B-it`), freeing ~110 GB per node
for the P1.003 memory probe and subsequent training.

Rationale: Full-parameter SFT of Qwen3.8-27B needs roughly 155 GiB across the two nodes. Each
node had 110–111 GB of its 121 GB held by Gemma. The two workloads cannot coexist. Shutdown was
deferred from 2026-08-16 until the weights were resident and a training harness existed, so the
endpoints were not lost during CPU-only setup — that deferral held for ~25 hours and cost nothing.

Alternatives Considered:
- Stop one node only — viable, since the two servers are independent (DEC-0006), and it would
  keep an endpoint alive. Rejected because full-parameter 27B training needs both nodes; this
  option would only work if the project re-scoped to LoRA.
- Defer further — rejected by Ramchand on 2026-08-17.

**Unresolved at the time of the decision, and recorded as such:** the question "what consumes
this endpoint, and is anything pointed at it?" was put to Ramchand and not answered before the
instruction to proceed. That is his call to make and it was made. It is noted here so that if
something downstream breaks, the cause is documented rather than discovered. `restart=no` means
nothing brings Gemma back automatically; restoring it is a deliberate act using the verified
command in `docs/RUNBOOK.md`.

Evidence / Source Docs: pre-shutdown state table in `tasks/atomic-task-list.md` (P1.005);
verified restore command in `docs/RUNBOOK.md`.
Measured Result: recorded post-shutdown in the same task entry.
Follow-up: P1.007 (build pinned image), then P1.003 (memory probe — settles Q006 and Q010).
Restore Gemma when Suvadu no longer needs exclusive use of the cluster.
Human Approval: **Approved by Ramchand 2026-08-16, re-affirmed and instructed to execute
2026-08-17.**
