"""Batch sources for training.

Two sources, and the distinction is deliberate:

**synthetic** — random token ids at the configured sequence length. This exists so TASK P1.003
can measure the memory profile of a real 27B optimizer step *before* any corpus is built. It is
the only way to answer "does full-parameter FT fit across two nodes" without first spending days
on data, and if the answer is no, the whole project re-scopes before that work happens.

**corpus** — the real thing. Not implemented: the corpus does not exist until G2, and writing a
loader against a schema that has not been frozen would be writing against a guess. Raises rather
than returning something plausible, per AGENTS.md §2.3.

A synthetic batch is *not* a substitute for a real one when measuring anything other than
memory. Loss on random tokens is meaningless, and throughput on uniformly-shaped batches
overstates throughput on a corpus with a realistic length distribution. Both caveats are
enforced by :func:`assert_not_synthetic_for_metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


class SyntheticDataForbidden(RuntimeError):
    """Raised when a synthetic-data run tries to produce a citable metric."""


@dataclass(frozen=True)
class BatchSpec:
    """Shape of one training batch."""

    batch_size: int
    seq_len: int
    vocab_size: int

    @property
    def tokens_per_batch(self) -> int:
        return self.batch_size * self.seq_len


def synthetic_batches(spec: BatchSpec, *, seed: int, n_batches: int) -> Iterator[dict]:
    """Yield deterministic random-token batches for memory profiling.

    Deterministic given the seed, so a memory probe is reproducible. Token ids avoid 0 to keep
    them clear of any pad id, since a batch that is accidentally mostly padding would understate
    activation memory — the exact quantity the probe exists to measure.
    """
    import torch  # deferred: provenance and planning must work on machines without torch

    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(n_batches):
        input_ids = torch.randint(
            low=1,
            high=spec.vocab_size,
            size=(spec.batch_size, spec.seq_len),
            generator=generator,
            dtype=torch.long,
        )
        yield {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def assert_not_synthetic_for_metrics(data_source: str, *, purpose: str) -> None:
    """Guard: synthetic data may profile memory, never produce a reported metric.

    Called at the point a run would write a metric. Without it, a memory probe left running
    with `--data synthetic` would emit a loss curve that looks like a training result.
    """
    if data_source == "synthetic":
        raise SyntheticDataForbidden(
            f"refusing to record {purpose!r} from synthetic data. Random tokens produce a "
            "meaningless loss and an optimistic throughput; only memory measurements are valid "
            "from this source (TASK P1.003)."
        )


def corpus_batches(*_args, **_kwargs):
    """Real corpus loader.

    Not implemented on purpose. The corpus is frozen at G2 (TASKs P2.004–P2.008) and its schema —
    chat-template rendering, reasoning-token handling, packing — is still open (Q007). A loader
    written now would encode a guess about a format that has not been decided.
    """
    raise NotImplementedError(
        "SUVADU-PLACEHOLDER: corpus loader awaits the G2 data freeze (P2.004-P2.008). "
        "Use --data synthetic for the P1.003 memory probe."
    )
