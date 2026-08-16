"""Provenance writer tests (TASK P0.003)."""

import json

import pytest

from suvadu.provenance import (
    MANIFEST_FILENAME,
    REQUIRED_IDENTIFIERS,
    ProvenanceError,
    canonical_hash,
    capture_environment,
    hash_file,
    validate_manifest,
    write_manifest,
)


def _manifest_payload(**overrides):
    payload = {
        "run_id": "phase0-test-001",
        "config_hash": "a" * 64,
        "code_sha": "b" * 40,
        "data_hash": "c" * 64,
        "seed": 42,
        "environment": {"python": "3.12.0", "machine": "aarch64"},
        "created_at": "2026-08-16T19:15:00+00:00",
        "git_dirty": False,
        "notes": {},
    }
    payload.update(overrides)
    return payload


def test_canonical_hash_is_key_order_independent():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_canonical_hash_distinguishes_values():
    assert canonical_hash({"lr": 1e-5}) != canonical_hash({"lr": 2e-5})


def test_hash_file_matches_known_digest(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_bytes(b"hello")
    # sha256("hello")
    assert hash_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_capture_environment_records_machine_and_python():
    env = capture_environment()
    assert env["python"]
    assert env["machine"]
    # Must not raise on a box with no torch — provenance has to work CPU-side too.
    assert "cuda_available" in env


def test_write_manifest_creates_all_five_identifiers(tmp_path, monkeypatch):
    monkeypatch.setattr("suvadu.provenance.git_state", lambda root=None: ("d" * 40, False))
    path = write_manifest(
        tmp_path / "run",
        run_id="phase0-test-001",
        config={"lr": 1e-5, "seed": 42},
        data_hash="c" * 64,
        seed=42,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name in REQUIRED_IDENTIFIERS:
        assert payload.get(name), f"{name} missing or empty"


def test_write_manifest_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr("suvadu.provenance.git_state", lambda root=None: ("d" * 40, False))
    kwargs = dict(run_id="phase0-test-001", config={}, data_hash="c" * 64, seed=1)
    write_manifest(tmp_path / "run", **kwargs)
    with pytest.raises(ProvenanceError, match="never reused"):
        write_manifest(tmp_path / "run", **kwargs)


def test_validate_manifest_roundtrip(tmp_path):
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    manifest = validate_manifest(tmp_path)
    assert manifest.run_id == "phase0-test-001"
    assert manifest.seed == 42


def test_validate_manifest_rejects_missing_directory(tmp_path):
    with pytest.raises(ProvenanceError) as exc:
        validate_manifest(tmp_path / "nope")
    assert set(exc.value.missing) == set(REQUIRED_IDENTIFIERS)


@pytest.mark.parametrize("identifier", REQUIRED_IDENTIFIERS)
def test_validate_manifest_names_each_missing_identifier(tmp_path, identifier):
    """An empty value counts as missing — a manifest that *looks* complete is worse than none."""
    empty = {"seed": None, "environment": {}}.get(identifier, "")
    payload = _manifest_payload(**{identifier: empty})
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProvenanceError) as exc:
        validate_manifest(tmp_path)
    assert identifier in exc.value.missing


def test_validate_manifest_rejects_malformed_json(tmp_path):
    (tmp_path / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="not valid JSON"):
        validate_manifest(tmp_path)
