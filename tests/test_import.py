"""Layout and import smoke tests (TASK P0.002)."""

from pathlib import Path

import suvadu

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_and_exposes_version():
    assert suvadu.__version__
    assert isinstance(suvadu.__version__, str)


def test_expected_layout_exists():
    expected = [
        "CLAUDE.md",
        "AGENTS.md",
        "CODEX.md",
        "README.md",
        "TASKS.md",
        ".gitignore",
        "tasks/atomic-task-list.md",
        "docs/DECISION_LOG.md",
        "docs/EXPERIMENT_LOG.md",
        "docs/REPRODUCIBILITY.md",
        "docs/RISKS_AND_OPEN_QUESTIONS.md",
        "docs/RUNBOOK.md",
        "src/suvadu/__init__.py",
    ]
    missing = [p for p in expected if not (REPO_ROOT / p).exists()]
    assert not missing, f"missing scaffolding: {missing}"


def test_governance_files_are_byte_identical():
    """CLAUDE.md and AGENTS.md must never drift apart (AGENTS.md is the contract for Codex)."""
    claude = (REPO_ROOT / "CLAUDE.md").read_bytes()
    agents = (REPO_ROOT / "AGENTS.md").read_bytes()
    assert claude == agents, "CLAUDE.md and AGENTS.md have diverged"


def test_gate_review_exists_for_every_gate():
    for n in range(8):
        path = REPO_ROOT / "docs" / f"GATE_G{n}_REVIEW.md"
        assert path.exists(), f"missing {path.name}"


def test_no_deprecated_torch_dtype_in_source():
    """Global rule: dtype=, never torch_dtype=."""
    offenders = [
        p for p in (REPO_ROOT / "src").rglob("*.py")
        # The config module names the deprecated key in order to reject it; that is the one
        # legitimate mention, and it lives behind an explicit raise.
        if "torch_dtype" in p.read_text(encoding="utf-8") and p.name != "config.py"
    ]
    assert not offenders, f"torch_dtype found in {offenders}"


def test_no_unfilled_placeholder_sentinels_in_source():
    offenders = [
        p for p in (REPO_ROOT / "src").rglob("*.py")
        if "SUVADU-PLACEHOLDER" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"placeholder sentinel present in reachable code: {offenders}"
