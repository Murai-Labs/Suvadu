"""Resume-state tests (TASK P1.004)."""

import json

import pytest

from suvadu.train.resume import RESUME_FILENAME, ResumeMismatch, ResumeState

IDS = dict(run_id="phase5-main-001", config_hash="a" * 64, data_hash="b" * 64, seed=42)


def make_state(**overrides):
    kwargs = dict(IDS, epoch=2, global_step=1500)
    kwargs.update(overrides)
    return ResumeState.create(**kwargs)


def test_write_then_read_roundtrip(tmp_path):
    make_state().write(tmp_path)
    got = ResumeState.read(tmp_path)
    assert got is not None
    assert got.epoch == 2
    assert got.global_step == 1500
    assert got.seed == 42


def test_read_returns_none_when_nothing_to_resume(tmp_path):
    assert ResumeState.read(tmp_path) is None


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    make_state().write(tmp_path)
    assert (tmp_path / RESUME_FILENAME).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_write_produces_valid_json(tmp_path):
    p = make_state().write(tmp_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["run_id"] == "phase5-main-001"
    assert payload["written_at"]


def test_validate_passes_on_identical_experiment(tmp_path):
    make_state().write(tmp_path)
    ResumeState.read(tmp_path).validate_against(**IDS)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("config_hash", "c" * 64),
        ("data_hash", "d" * 64),
        ("seed", 7),
        ("run_id", "phase5-main-002"),
    ],
)
def test_validate_rejects_each_changed_identifier(tmp_path, field, bad):
    make_state().write(tmp_path)
    requested = dict(IDS)
    requested[field] = bad
    with pytest.raises(ResumeMismatch) as exc:
        ResumeState.read(tmp_path).validate_against(**requested)
    assert field in exc.value.differences


def test_validate_reports_all_differences_at_once(tmp_path):
    """One failed launch should reveal every problem, not one per attempt."""
    make_state().write(tmp_path)
    with pytest.raises(ResumeMismatch) as exc:
        ResumeState.read(tmp_path).validate_against(
            run_id="phase5-main-002", config_hash="c" * 64, data_hash="d" * 64, seed=7
        )
    assert set(exc.value.differences) == {"run_id", "config_hash", "data_hash", "seed"}


def test_mismatch_message_names_stored_and_requested(tmp_path):
    make_state().write(tmp_path)
    requested = dict(IDS, seed=7)
    with pytest.raises(ResumeMismatch, match="stored=42 requested=7"):
        ResumeState.read(tmp_path).validate_against(**requested)


def test_overwriting_advances_progress(tmp_path):
    """Resume state is rewritten each epoch; unlike the manifest, it is meant to be replaced."""
    make_state(epoch=1, global_step=750).write(tmp_path)
    make_state(epoch=2, global_step=1500).write(tmp_path)
    assert ResumeState.read(tmp_path).global_step == 1500
