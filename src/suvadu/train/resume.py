"""Resume state, and the checks that stop a resume from silently becoming a new experiment.

AGENTS.md §4: runs over 30 minutes write resume state each epoch and validate config and seed on
restart. The failure this prevents is specific and quiet — a job dies at epoch 2, someone edits
the config, restarts with `--resume`, and the run completes carrying two different
hyperparameter sets under one run id. The metrics look fine. The manifest looks fine. The
experiment is meaningless and nothing in the artifacts says so.

So a mismatch is a hard failure, never a warning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

RESUME_FILENAME = "resume_state.json"


class ResumeMismatch(RuntimeError):
    """Raised when resume state does not match the config being resumed with.

    Carries the specific fields that differ, because "config changed" is not actionable and
    "seed 42 -> 7" is.
    """

    def __init__(self, message: str, *, differences: dict[str, tuple[object, object]]) -> None:
        super().__init__(message)
        self.differences = differences


@dataclass(frozen=True)
class ResumeState:
    """Everything needed to prove a resume continues the same experiment."""

    run_id: str
    config_hash: str
    data_hash: str
    seed: int
    epoch: int
    global_step: int
    written_at: str

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        config_hash: str,
        data_hash: str,
        seed: int,
        epoch: int,
        global_step: int,
    ) -> "ResumeState":
        return cls(
            run_id=run_id,
            config_hash=config_hash,
            data_hash=data_hash,
            seed=int(seed),
            epoch=int(epoch),
            global_step=int(global_step),
            written_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def write(self, run_dir: str | Path) -> Path:
        """Write atomically.

        Via a temp file and replace: a crash partway through writing resume state would
        otherwise leave truncated JSON, turning a recoverable interruption into a lost run.
        """
        path = Path(run_dir) / RESUME_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def read(cls, run_dir: str | Path) -> "ResumeState | None":
        """Return the stored state, or None if there is nothing to resume from."""
        path = Path(run_dir) / RESUME_FILENAME
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})

    def validate_against(
        self, *, run_id: str, config_hash: str, data_hash: str, seed: int
    ) -> None:
        """Raise ResumeMismatch unless this state describes the same experiment.

        All four identifiers are checked together rather than short-circuiting, so one restart
        attempt reports every problem instead of revealing them one failed launch at a time.
        """
        expected = {
            "run_id": run_id,
            "config_hash": config_hash,
            "data_hash": data_hash,
            "seed": int(seed),
        }
        actual = {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "seed": self.seed,
        }
        diffs = {k: (actual[k], expected[k]) for k in expected if actual[k] != expected[k]}
        if diffs:
            detail = "; ".join(f"{k}: stored={a!r} requested={b!r}" for k, (a, b) in diffs.items())
            raise ResumeMismatch(
                "refusing to resume: the stored run does not match the requested one "
                f"({detail}). Resuming across a changed config would produce one run id "
                "carrying two experiments. Allocate a new run id instead.",
                differences=diffs,
            )
