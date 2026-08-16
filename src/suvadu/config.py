"""Versioned config contract.

AGENTS.md §2.4: configurations are versioned, locked and traceable. This module exists to make
config drift *loud*. Every rule below rejects a class of silent failure that would otherwise
show up as an uninterpretable results table weeks later.

Note on dtype: the field is ``dtype``. HuggingFace's ``torch_dtype`` is deprecated and is
rejected outright by :func:`RunConfig.from_dict` rather than quietly accepted — a config that
sets ``torch_dtype`` would be silently ignored by this loader, which is exactly the kind of
no-op setting that wastes a training run.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

SCHEMA_VERSION = 1

Regime = Literal["full_ft", "lora"]

ALLOWED_REGIMES: tuple[str, ...] = ("full_ft", "lora")
ALLOWED_DTYPES: tuple[str, ...] = ("bfloat16", "float32")

#: Revision strings that are not reproducible. A moving pointer is not a pinned input.
UNPINNED_REVISIONS: tuple[str, ...] = ("main", "master", "latest", "HEAD", "")


class ConfigError(ValueError):
    """Raised when a config violates the contract. Names the offending field."""

    def __init__(self, message: str, *, fieldname: str | None = None) -> None:
        super().__init__(message)
        self.fieldname = fieldname


@dataclass(frozen=True)
class RunConfig:
    """One arm of the experiment.

    An "arm" is a single training or evaluation configuration — the treatment, or one of the
    baselines B1–B4. Arms are compared to each other, so fields that must match across arms
    (everything except ``corpus_path`` for the B1 comparison) are all captured here, in one
    object, so a diff of two configs is a complete statement of how two arms differ.
    """

    # --- identity -------------------------------------------------------------------
    schema_version: int
    arm: str
    run_id: str

    # --- model ----------------------------------------------------------------------
    base_model: str
    base_revision: str
    dtype: str

    # --- data -----------------------------------------------------------------------
    corpus_path: str
    corpus_hash: str
    max_seq_len: int

    # --- optimisation ---------------------------------------------------------------
    regime: str
    seed: int
    epochs: int
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int

    # --- optional -------------------------------------------------------------------
    gradient_checkpointing: bool = True
    optimizer: str = "adamw_8bit"
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_target_modules: tuple[str, ...] = ()
    log_every_n_steps: int = 100
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ConfigError(
                f"schema_version {self.schema_version} != {SCHEMA_VERSION}; refusing to guess "
                "how to migrate an unknown schema.",
                fieldname="schema_version",
            )

        if self.regime not in ALLOWED_REGIMES:
            raise ConfigError(
                f"regime must be one of {ALLOWED_REGIMES}, got {self.regime!r}",
                fieldname="regime",
            )

        if self.dtype not in ALLOWED_DTYPES:
            raise ConfigError(
                f"dtype must be one of {ALLOWED_DTYPES}, got {self.dtype!r}",
                fieldname="dtype",
            )

        if self.base_revision in UNPINNED_REVISIONS:
            raise ConfigError(
                f"base_revision {self.base_revision!r} is a moving pointer, not a pinned "
                "revision. Pin the commit SHA — an unpinned base model makes every result in "
                "this project irreproducible.",
                fieldname="base_revision",
            )

        if not self.corpus_hash:
            raise ConfigError(
                "corpus_hash is required. The corpus is regenerated from a live machine whose "
                "session traces keep accumulating; without a hash, two different datasets are "
                "indistinguishable in a results table.",
                fieldname="corpus_hash",
            )

        if self.log_every_n_steps > 100:
            raise ConfigError(
                f"log_every_n_steps={self.log_every_n_steps} exceeds 100. Long runs must emit "
                "progress at least every 100 steps (AGENTS.md 4) — a silent job is "
                "indistinguishable from a hang.",
                fieldname="log_every_n_steps",
            )

        for name in ("epochs", "max_seq_len", "per_device_batch_size",
                     "gradient_accumulation_steps", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ConfigError(f"{name} must be a non-negative int, got {value!r}",
                                  fieldname=name)

        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be positive, got {self.learning_rate!r}",
                              fieldname="learning_rate")

        if self.regime == "lora":
            if not self.lora_rank or not self.lora_alpha:
                raise ConfigError(
                    "regime='lora' requires lora_rank and lora_alpha.",
                    fieldname="lora_rank",
                )
            if not self.lora_target_modules:
                raise ConfigError(
                    "regime='lora' requires an explicit lora_target_modules list. Qwen3.8-27B is "
                    "a hybrid: 16 of 64 layers are attention and 48 are Gated DeltaNet, so a "
                    "default preset does not mean here what it means on a dense transformer. "
                    "The choice must be deliberate and recorded (see Q009).",
                    fieldname="lora_target_modules",
                )
        else:
            if self.lora_rank or self.lora_alpha or self.lora_target_modules:
                raise ConfigError(
                    f"regime={self.regime!r} but LoRA fields are set. A config that carries "
                    "settings its regime ignores is a silent no-op waiting to happen.",
                    fieldname="lora_rank",
                )

    # ------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunConfig":
        """Build from a plain dict, rejecting unknown and deprecated keys."""
        if "torch_dtype" in payload:
            raise ConfigError(
                "torch_dtype is deprecated and is not read by this loader; use dtype=. Leaving "
                "it in place would silently have no effect.",
                fieldname="torch_dtype",
            )

        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(
                f"unknown config key(s): {', '.join(sorted(unknown))}. Unknown keys are rejected "
                "rather than ignored, because an ignored key looks like a setting that took "
                "effect.",
                fieldname=sorted(unknown)[0],
            )

        missing = {
            f.name for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING  # type: ignore[misc]
        } - set(payload)
        if missing:
            raise ConfigError(f"missing required config key(s): {', '.join(sorted(missing))}")

        data = dict(payload)
        if "lora_target_modules" in data and data["lora_target_modules"] is not None:
            data["lora_target_modules"] = tuple(data["lora_target_modules"])
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ConfigError(f"{path} did not parse to a mapping")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def diff(self, other: "RunConfig") -> dict[str, tuple[Any, Any]]:
        """Fields that differ between two arms.

        Used at G4: the B1 control must differ from the treatment in ``corpus_path`` and
        ``corpus_hash`` (and its identity fields) and in nothing else. This method is how that
        claim is checked rather than asserted.
        """
        mine, theirs = self.to_dict(), other.to_dict()
        return {k: (mine[k], theirs[k]) for k in mine if mine[k] != theirs[k]}
