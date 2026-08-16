"""Run provenance: the five identifiers without which no metric is citable.

AGENTS.md §2.4 and §2.6 make this a precondition for running, not a post-hoc record. The writer
is called *before* the first training step; a run directory that fails :func:`validate_manifest`
produces results that go to ``notes/untrusted-results.md`` and are excluded from the report and
the model card.

The five identifier categories are fixed and are not configurable:

    config_hash   sha256 of the canonicalised config
    code_sha      git commit of the working tree, plus whether it was dirty
    data_hash     sha256 (or recorded split id) of the corpus actually consumed
    seed          the integer seed
    environment   machine, python, and — when available — torch/CUDA/GPU details

``data_hash`` matters more here than on a typical project: the corpus is regenerated from a live
machine whose session traces keep accumulating, so two exports a week apart are different
datasets that would otherwise be indistinguishable in a results table.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "manifest.json"

#: The identifier categories every manifest must carry. Fixed by contract.
REQUIRED_IDENTIFIERS = ("config_hash", "code_sha", "data_hash", "seed", "environment")


class ProvenanceError(Exception):
    """Raised when a run directory's provenance is missing or malformed.

    Carries the specific missing identifiers so the caller does not have to guess which of the
    five failed.
    """

    def __init__(self, message: str, *, missing: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.missing = missing


@dataclass(frozen=True)
class RunManifest:
    """The immutable provenance record for one run."""

    run_id: str
    config_hash: str
    code_sha: str
    data_hash: str
    seed: int
    environment: dict[str, Any]
    created_at: str
    git_dirty: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def canonical_hash(obj: Any) -> str:
    """sha256 of a canonically-serialised object.

    Canonical means sorted keys and no insignificant whitespace, so two configs that differ only
    in key order hash identically — otherwise a cosmetic reordering would look like a different
    experiment.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """sha256 of a file, streamed.

    Chunked because the corpus files are hundreds of megabytes and must not be read into memory
    to be hashed.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo_root: str | Path | None = None) -> tuple[str, bool]:
    """Return ``(commit_sha, is_dirty)`` for the working tree.

    A dirty tree is recorded rather than rejected: refusing to run would be worse than running
    with the fact written down. But the flag travels with every metric the run produces, because
    a commit SHA alone does not reproduce a dirty tree.

    Raises ProvenanceError when git is unavailable or the path is not a repository — a run whose
    code version cannot be identified has no provenance.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ProvenanceError(
            f"cannot determine git state for {root}: {exc}. A run without an identifiable code "
            "version cannot be provenanced (AGENTS.md 2.4).",
            missing=("code_sha",),
        ) from exc
    return sha, bool(status)


def capture_environment() -> dict[str, Any]:
    """Record the execution environment.

    Deliberately import-guarded around torch: this must work on a login shell with no CUDA as
    well as inside the training container, and an ImportError here would block provenance for
    CPU-side tasks that legitimately have no GPU.
    """
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }

    for var in ("CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR"):
        if var in os.environ:
            env[f"env.{var}"] = os.environ[var]

    try:
        import torch  # noqa: PLC0415  (deliberately deferred; see docstring)
    except ImportError:
        env["torch"] = None
        env["cuda_available"] = False
        return env

    env["torch"] = torch.__version__
    env["cuda_available"] = bool(torch.cuda.is_available())
    if env["cuda_available"]:
        env["cuda"] = torch.version.cuda
        env["device_count"] = torch.cuda.device_count()
        env["device_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        # Recorded as a string so "sm_121" is greppable in results; GB10 is sm_121.
        env["device_capability"] = f"sm_{major}{minor}"

    try:
        import transformers  # noqa: PLC0415
    except ImportError:
        env["transformers"] = None
    else:
        env["transformers"] = transformers.__version__

    return env


def write_manifest(
    run_dir: str | Path,
    *,
    run_id: str,
    config: Any,
    data_hash: str,
    seed: int,
    repo_root: str | Path | None = None,
    notes: dict[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` into ``run_dir`` and return its path.

    Call this *before* the first step. Refuses to overwrite an existing manifest, because a run
    id is never reused (AGENTS.md §8) and silently replacing one would destroy the record of a
    previous attempt.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    manifest_path = run_path / MANIFEST_FILENAME

    if manifest_path.exists():
        raise ProvenanceError(
            f"{manifest_path} already exists. Run ids are never reused, even for failed "
            "attempts — allocate a new one (AGENTS.md 8)."
        )

    code_sha, dirty = git_state(repo_root)

    manifest = RunManifest(
        run_id=run_id,
        config_hash=canonical_hash(config),
        code_sha=code_sha,
        data_hash=data_hash,
        seed=int(seed),
        environment=capture_environment(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_dirty=dirty,
        notes=notes or {},
    )

    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    return manifest_path


def validate_manifest(run_dir: str | Path) -> RunManifest:
    """Load and validate a run's manifest.

    Raises ProvenanceError naming exactly which identifiers are missing or empty. An empty string
    counts as missing: a manifest with ``"data_hash": ""`` is worse than no manifest, because it
    looks complete.
    """
    manifest_path = Path(run_dir) / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ProvenanceError(
            f"no {MANIFEST_FILENAME} in {run_dir}; results from this run are untrusted "
            "(AGENTS.md 2.6).",
            missing=REQUIRED_IDENTIFIERS,
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"{manifest_path} is not valid JSON: {exc}") from exc

    missing = tuple(
        name for name in REQUIRED_IDENTIFIERS
        if payload.get(name) in (None, "", {}, [])
    )
    if missing:
        raise ProvenanceError(
            f"{manifest_path} is missing required identifier(s): {', '.join(missing)}. "
            "No metric from this run is citable.",
            missing=missing,
        )

    known = set(RunManifest.__dataclass_fields__)
    return RunManifest(**{k: v for k, v in payload.items() if k in known})
