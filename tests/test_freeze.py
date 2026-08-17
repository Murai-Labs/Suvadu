"""Freeze-policy tests (TASK P1.004).

Parameter names below are the real prefixes from Qwen3.8-27B's safetensors index, read
2026-08-16 — not invented examples.
"""

import pytest

from suvadu.train.freeze import (
    DEFAULT_POLICY,
    FREEZE_POLICIES,
    ParamGroup,
    UnknownFreezePolicy,
    classify_parameter,
    summarise_trainable,
)

REAL_NAMES = [
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.linear_attn.A_log",
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "model.language_model.layers.3.self_attn.o_proj.weight",
    "model.language_model.norm.weight",
    "model.visual.blocks.0.attn.qkv.weight",
    "model.visual.blocks.0.mlp.linear_fc1.weight",
    "model.visual.patch_embed.proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.fc.weight",
    "mtp.norm.weight",
    "lm_head.weight",
]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", ParamGroup.LANGUAGE_MODEL),
        ("model.visual.blocks.0.attn.qkv.weight", ParamGroup.VISION),
        ("mtp.fc.weight", ParamGroup.MTP),
        ("lm_head.weight", ParamGroup.LM_HEAD),
        ("some.unexpected.tensor", ParamGroup.OTHER),
    ],
)
def test_classify_parameter(name, expected):
    assert classify_parameter(name) is expected


def test_default_policy_freezes_vision_and_mtp():
    s = summarise_trainable(REAL_NAMES, DEFAULT_POLICY)
    assert s.frozen_by_group.get("vision") == 3
    assert s.frozen_by_group.get("mtp") == 3
    assert "language_model" not in s.frozen_by_group
    assert "lm_head" not in s.frozen_by_group


def test_vision_only_policy_keeps_mtp_trainable():
    s = summarise_trainable(REAL_NAMES, "vision_only")
    assert s.frozen_by_group.get("vision") == 3
    assert "mtp" not in s.frozen_by_group
    assert s.counts_by_group.get("mtp") == 3


def test_none_policy_trains_everything():
    s = summarise_trainable(REAL_NAMES, "none")
    assert s.n_frozen == 0
    assert s.n_trainable == len(REAL_NAMES)


def test_counts_are_exhaustive():
    """Every parameter is either trained or frozen — none silently unaccounted for."""
    for policy in FREEZE_POLICIES:
        s = summarise_trainable(REAL_NAMES, policy)
        assert s.n_total == len(REAL_NAMES), policy
        assert sum(s.counts_by_group.values()) == s.n_trainable, policy
        assert sum(s.frozen_by_group.values()) == s.n_frozen, policy


def test_unknown_policy_raises_before_doing_anything():
    with pytest.raises(UnknownFreezePolicy, match="unknown freeze policy"):
        summarise_trainable(REAL_NAMES, "freeze_evrything")


CAUSALLM_NAMES = [
    # AutoModelForCausalLM instantiates the text tower UNWRAPPED. Verified on a meta device
    # 2026-08-17: 851 tensors, 26,895,998,464 params, zero vision, zero MTP.
    "model.embed_tokens.weight",
    "model.layers.0.linear_attn.dt_bias",
    "model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.layers.3.self_attn.q_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
]


def test_causallm_naming_is_classified_as_language_model():
    """Regression guard for the 2026-08-17 near-miss.

    AutoModelForCausalLM names params `model.layers.*`, not `model.language_model.layers.*`.
    An earlier classifier keyed only on the wrapped form and filed all 850 language tensors as
    OTHER. Every policy trains OTHER, so the trainable count looked correct and only the group
    breakdown revealed it.
    """
    for name in CAUSALLM_NAMES:
        if name.startswith("lm_head"):
            continue
        assert classify_parameter(name) is ParamGroup.LANGUAGE_MODEL, name

    s = summarise_trainable(CAUSALLM_NAMES, DEFAULT_POLICY)
    assert s.counts_by_group.get("language_model") == 5
    assert s.counts_by_group.get("lm_head") == 1
    assert "other" not in s.counts_by_group, "text-only names must not fall through to OTHER"


def test_both_naming_schemes_agree_on_group_membership():
    """The same weight must classify identically whether wrapped or unwrapped."""
    pairs = [
        ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
         "model.layers.0.linear_attn.in_proj_qkv.weight"),
        ("model.language_model.norm.weight", "model.norm.weight"),
    ]
    for wrapped, unwrapped in pairs:
        assert classify_parameter(wrapped) is classify_parameter(unwrapped) is ParamGroup.LANGUAGE_MODEL


def test_recurrent_layers_are_language_model_not_other():
    """The 48 linear_attention layers must be trainable under the default policy.

    Guards Q009: these expose in_proj_a/b/qkv/z + out_proj rather than q_proj/k_proj/v_proj,
    and a classifier keyed on the conventional names would drop them into OTHER.
    """
    recurrent = [
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.dt_bias",
    ]
    for n in recurrent:
        assert classify_parameter(n) is ParamGroup.LANGUAGE_MODEL, n
    s = summarise_trainable(recurrent, DEFAULT_POLICY)
    assert s.n_trainable == len(recurrent)
    assert s.n_frozen == 0
