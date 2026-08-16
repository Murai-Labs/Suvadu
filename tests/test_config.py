"""Config contract tests (TASK P0.004)."""

import pytest

from suvadu.config import SCHEMA_VERSION, ConfigError, RunConfig


def base_payload(**overrides):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "arm": "treatment",
        "run_id": "phase4-treatment-001",
        "base_model": "Qwen/Qwen3.8-27B",
        "base_revision": "0f1e2d3c4b5a69788796a5b4c3d2e1f0a1b2c3d4",
        "dtype": "bfloat16",
        "corpus_path": "/data/suvadu/treatment.jsonl",
        "corpus_hash": "e" * 64,
        "max_seq_len": 8192,
        "regime": "full_ft",
        "seed": 42,
        "epochs": 3,
        "learning_rate": 1e-5,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 16,
    }
    payload.update(overrides)
    return payload


def test_valid_config_loads():
    cfg = RunConfig.from_dict(base_payload())
    assert cfg.arm == "treatment"
    assert cfg.regime == "full_ft"


def test_rejects_deprecated_torch_dtype():
    with pytest.raises(ConfigError, match="torch_dtype is deprecated"):
        RunConfig.from_dict(base_payload(torch_dtype="bfloat16"))


def test_rejects_unknown_key():
    with pytest.raises(ConfigError, match="unknown config key"):
        RunConfig.from_dict(base_payload(lr_schedule="cosine"))


def test_rejects_missing_required_key():
    payload = base_payload()
    del payload["corpus_hash"]
    with pytest.raises(ConfigError, match="missing required config key"):
        RunConfig.from_dict(payload)


@pytest.mark.parametrize("revision", ["main", "master", "latest", "HEAD", ""])
def test_rejects_unpinned_base_revision(revision):
    with pytest.raises(ConfigError, match="moving pointer"):
        RunConfig.from_dict(base_payload(base_revision=revision))


def test_rejects_empty_corpus_hash():
    with pytest.raises(ConfigError, match="corpus_hash is required"):
        RunConfig.from_dict(base_payload(corpus_hash=""))


def test_rejects_wrong_schema_version():
    with pytest.raises(ConfigError, match="schema_version"):
        RunConfig.from_dict(base_payload(schema_version=SCHEMA_VERSION + 1))


def test_rejects_bad_dtype():
    with pytest.raises(ConfigError, match="dtype must be one of"):
        RunConfig.from_dict(base_payload(dtype="float16"))


def test_rejects_infrequent_logging():
    """AGENTS.md 4: progress at least every 100 steps."""
    with pytest.raises(ConfigError, match="exceeds 100"):
        RunConfig.from_dict(base_payload(log_every_n_steps=500))


def test_lora_requires_explicit_target_modules():
    with pytest.raises(ConfigError, match="explicit lora_target_modules"):
        RunConfig.from_dict(base_payload(regime="lora", lora_rank=32, lora_alpha=64))


def test_lora_requires_rank_and_alpha():
    with pytest.raises(ConfigError, match="requires lora_rank"):
        RunConfig.from_dict(
            base_payload(regime="lora", lora_target_modules=["q_proj", "v_proj"])
        )


def test_full_ft_rejects_stray_lora_fields():
    """A setting its regime ignores is a silent no-op waiting to happen."""
    with pytest.raises(ConfigError, match="LoRA fields are set"):
        RunConfig.from_dict(base_payload(regime="full_ft", lora_rank=32))


def test_valid_lora_config_loads():
    cfg = RunConfig.from_dict(
        base_payload(
            arm="B4",
            regime="lora",
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    )
    assert cfg.lora_target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_diff_isolates_the_control_variable():
    """The B1 control must differ from the treatment ONLY in corpus + identity fields.

    This is the check that makes G4 adjudicable: if the diff contains a hyperparameter, the two
    arms are not token-matched and the comparison is invalid.
    """
    treatment = RunConfig.from_dict(base_payload())
    control = RunConfig.from_dict(
        base_payload(
            arm="B1",
            run_id="phase4-baseline-b1-001",
            corpus_path="/data/suvadu/control.jsonl",
            corpus_hash="f" * 64,
        )
    )
    differing = set(treatment.diff(control))
    assert differing == {"arm", "run_id", "corpus_path", "corpus_hash"}


def test_from_yaml_roundtrip(tmp_path):
    import yaml

    path = tmp_path / "arm.yaml"
    path.write_text(yaml.safe_dump(base_payload()), encoding="utf-8")
    cfg = RunConfig.from_yaml(path)
    assert cfg.run_id == "phase4-treatment-001"
