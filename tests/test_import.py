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


def test_placeholder_sentinels_are_always_raises():
    """AGENTS.md 2.3 prescribes the sentinel; what it forbids is a *silent* placeholder.

    So the check is not "no sentinel exists" — an earlier version of this test asserted that and
    was wrong, since it outlawed the mechanism the contract mandates. The check is that every
    sentinel sits inside a raise, never beside a return. A `return 0.0  # TODO` is the failure
    mode; `raise NotImplementedError("SUVADU-PLACEHOLDER: ...")` is the required alternative.

    The pre-run gate is separate and lives in docs/RUNBOOK.md: before any run,
    `grep -r SUVADU-PLACEHOLDER src` must be empty *for code that run reaches*.
    """
    bad = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if "SUVADU-PLACEHOLDER" not in line:
                continue
            window = " ".join(lines[max(0, i - 3):i + 1])
            if "raise NotImplementedError" not in window:
                bad.append(f"{path.name}:{i + 1}")
    assert not bad, f"placeholder sentinel not inside a raise: {bad}"


def test_placeholder_inventory_is_known():
    """Enumerate every deferred code path, so none is forgotten before a gate.

    Update this list deliberately when a placeholder is implemented or added.
    """
    found = {
        path.name
        for path in (REPO_ROOT / "src").rglob("*.py")
        if "SUVADU-PLACEHOLDER" in path.read_text(encoding="utf-8")
    }
    # data.py: corpus loader, blocked on the G2 data freeze.
    assert found == {"data.py"}, f"placeholder inventory changed: {sorted(found)}"
