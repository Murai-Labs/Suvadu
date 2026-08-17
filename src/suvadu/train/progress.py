"""Progress reporting for long runs.

AGENTS.md §4 and the standing lab rule: anything over ~30 s emits `step/total`, elapsed and ETA
at least every 100 steps. The rule exists because a Runpod run once went 27 minutes silent and
had to be asked whether it was alive — a silent job is indistinguishable from a hang, an OOM, or
a stalled GPU. A JSON file written each epoch is not progress.

This module is pure formatting and arithmetic with no torch dependency, so the thing that is
supposed to tell you a run is alive is itself covered by tests that run on a laptop.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, TextIO


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration. Hours matter here — main runs are measured in tens of hours."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@dataclass
class ProgressReporter:
    """Emits a progress line every `log_every` steps, and always on the final step.

    `clock` is injectable so the ETA arithmetic can be tested deterministically rather than by
    sleeping.
    """

    total_steps: int
    log_every: int = 100
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    clock: Callable[[], float] = time.monotonic
    prefix: str = "[suvadu]"

    _start: float = field(init=False, default=0.0)
    _started: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if self.log_every <= 0 or self.log_every > 100:
            # >100 would violate the contract; <=0 would divide by zero. Both are config bugs
            # worth failing on rather than clamping silently.
            raise ValueError(
                f"log_every must be in 1..100 (AGENTS.md 4), got {self.log_every}"
            )

    def start(self) -> None:
        self._start = self.clock()
        self._started = True

    def should_log(self, step: int) -> bool:
        """Log on the first step, every `log_every` steps, and on the last step.

        The first step is included because the most valuable progress line is the one that
        proves the loop entered at all.
        """
        return step == 1 or step % self.log_every == 0 or step >= self.total_steps

    def elapsed(self) -> float:
        if not self._started:
            raise RuntimeError("ProgressReporter.start() was never called")
        return self.clock() - self._start

    def eta_seconds(self, step: int) -> float:
        """Linear ETA from mean throughput so far. Returns 0.0 at completion."""
        if step <= 0:
            return 0.0
        remaining = max(0, self.total_steps - step)
        if remaining == 0:
            return 0.0
        return (self.elapsed() / step) * remaining

    def format_line(
        self,
        step: int,
        *,
        loss: float | None = None,
        tokens_per_s: float | None = None,
        peak_mem_gib: float | None = None,
        extra: str = "",
    ) -> str:
        el = self.elapsed()
        pct = 100.0 * step / self.total_steps
        parts = [
            f"{self.prefix} step {step}/{self.total_steps} ({pct:5.1f}%)",
            f"elapsed {_fmt_duration(el)}",
            f"eta {_fmt_duration(self.eta_seconds(step))}",
        ]
        if loss is not None:
            parts.append(f"loss {loss:.4f}")
        if tokens_per_s is not None:
            parts.append(f"tok/s {tokens_per_s:,.0f}")
        if peak_mem_gib is not None:
            parts.append(f"peak {peak_mem_gib:.1f}GiB")
        if extra:
            parts.append(extra)
        return "  ".join(parts)

    def log(self, step: int, **kwargs) -> bool:
        """Emit a line if this step warrants one. Returns whether it emitted."""
        if not self.should_log(step):
            return False
        print(self.format_line(step, **kwargs), file=self.stream, flush=True)
        return True
