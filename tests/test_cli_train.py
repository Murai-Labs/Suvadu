"""CLI tests (TASK P1.004).

Only `--plan` is exercised here: it is the whole path that must work without a GPU, without
weights and without torch, which is exactly the part worth testing on a laptop. The training
path needs two GB10 nodes and is verified by P1.003, not by unit tests pretending to be one.
"""

from pathlib import Path

import pytest

from suvadu.cli.train import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = str(REPO_ROOT / "configs" / "phase1" / "smoke.yaml")


def test_smoke_config_is_valid_and_plans(capsys):
    assert main(["--config", SMOKE, "--plan"]) == 0
    out = capsys.readouterr().out
    assert "arm='smoke'" in out
    assert "Qwen/Qwen3.8-27B@1d4bf0f2ff60" in out
    assert "config_hash" in out


def test_plan_warns_loudly_about_synthetic_data(capsys):
    main(["--config", SMOKE, "--plan", "--data", "synthetic"])
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out
    assert "metrics will be refused" in out


def test_plan_resolves_freeze_policy_against_real_names(tmp_path, capsys):
    names = tmp_path / "names.txt"
    names.write_text(
        "\n".join(
            [
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                "model.language_model.layers.3.self_attn.q_proj.weight",
                "model.visual.blocks.0.attn.qkv.weight",
                "mtp.fc.weight",
                "lm_head.weight",
            ]
        ),
        encoding="utf-8",
    )
    main(["--config", SMOKE, "--plan", "--param-names", str(names)])
    out = capsys.readouterr().out
    assert "trainable 3" in out          # 2 language_model + lm_head
    assert "frozen    2" in out          # vision + mtp


def test_invalid_config_exits_2_without_loading_anything(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: 1\narm: x\nrun_id: r\nbase_model: m\n"
        "base_revision: main\ndtype: bfloat16\ncorpus_path: p\ncorpus_hash: h\n"
        "max_seq_len: 8\nregime: full_ft\nseed: 1\nepochs: 1\nlearning_rate: 1.0e-5\n"
        "per_device_batch_size: 1\ngradient_accumulation_steps: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(bad), "--plan"])
    assert exc.value.code == 2
    assert "moving pointer" in capsys.readouterr().err


def test_parser_defaults_to_corpus_not_synthetic():
    """Synthetic must be opt-in; a forgotten flag should never silently profile random tokens."""
    args = build_parser().parse_args(["--config", SMOKE])
    assert args.data == "corpus"
    assert args.plan is False
    assert args.resume is False
