"""Which parameters actually get trained.

Qwen3.8-27B is `Qwen3_5ForConditionalGeneration`: a language model, a vision tower, and a
multi-token-prediction head, all in one checkpoint. Measured from
`model.safetensors.index.json` on 2026-08-16:

    model.language_model.*   850 tensors
    model.visual.*           333 tensors
    mtp.*                     13 tensors
    lm_head.weight             1 tensor

Suvadu's corpus is text. Training the vision tower on it is not merely wasteful — every
trainable parameter costs a gradient *and* two optimizer-state bytes, and the 2-node fit is
already marginal (~155 GiB estimated against 2 × ~112.7 GiB usable). Freezing is the cheapest
lever available, and it is a decision that must be made explicitly rather than inherited from a
framework default. See `docs/decisions/0007-parameter-freezing.md`.

The MTP head is a subtler case. It ships in the checkpoint and drafts tokens for speculative
decoding, which is where the companion inference repo's ~2× decode speedup comes from. It drafts
for the *base* model. Fine-tune the target and leave the head frozen, and acceptance rate drifts
— the speedup degrades, though quality should hold because rejected drafts are resampled. Train
it on a text corpus and it at least stays aligned with the model it drafts for. Neither option is
free, so the policy is a config field, not a hardcoded choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ParamGroup(str, Enum):
    """Which sub-network a parameter belongs to."""

    LANGUAGE_MODEL = "language_model"
    VISION = "vision"
    MTP = "mtp"
    LM_HEAD = "lm_head"
    OTHER = "other"


#: policy name -> the groups it trains.
FREEZE_POLICIES: dict[str, frozenset[ParamGroup]] = {
    # Train everything. Included so the "no freezing" arm is nameable and comparable, not
    # because it is expected to fit.
    "none": frozenset(ParamGroup),
    # Default for a text-only SFT: language model + output head, nothing else.
    "vision_and_mtp": frozenset({ParamGroup.LANGUAGE_MODEL, ParamGroup.LM_HEAD,
                                 ParamGroup.OTHER}),
    # Freeze the vision tower only; keep the MTP head aligned with the updated target.
    "vision_only": frozenset({ParamGroup.LANGUAGE_MODEL, ParamGroup.LM_HEAD, ParamGroup.MTP,
                              ParamGroup.OTHER}),
}

DEFAULT_POLICY = "vision_and_mtp"


class UnknownFreezePolicy(ValueError):
    """Raised for a policy name that is not in FREEZE_POLICIES."""


#: Suffix-free prefixes that identify language-model parameters when the model is loaded
#: WITHOUT the multimodal wrapper. Verified on a meta device, 2026-08-17.
_TEXT_ONLY_PREFIXES = ("model.layers.", "model.embed_tokens", "model.norm", "layers.")


def classify_parameter(name: str) -> ParamGroup:
    """Map a parameter name to its sub-network.

    **Two naming schemes exist for the same weights**, and this cost a near-miss on 2026-08-17.
    The checkpoint and `AutoModelForImageTextToText` use the wrapped form::

        model.language_model.layers.0.linear_attn.in_proj_qkv.weight
        model.visual.blocks.0.attn.qkv.weight

    but `AutoModelForCausalLM` instantiates the text tower alone, unwrapped::

        model.layers.0.linear_attn.dt_bias
        model.embed_tokens.weight

    An earlier version keyed only on the wrapped form, so under `AutoModelForCausalLM` all 850
    language-model tensors fell through to ``OTHER``. Every current policy happens to train
    ``OTHER``, so the *outcome* was right and the *reasoning* was wrong — the failure was
    invisible in the trainable count and visible only in the group breakdown. That is precisely
    the kind of latent misclassification a future policy would silently inherit.
    """
    if name.startswith("mtp."):
        return ParamGroup.MTP
    if name.startswith("lm_head"):
        return ParamGroup.LM_HEAD
    if ".visual." in name or name.startswith("visual.") or name.startswith("model.visual."):
        return ParamGroup.VISION
    if name.startswith("model.language_model.") or name.startswith("language_model."):
        return ParamGroup.LANGUAGE_MODEL
    if name.startswith(_TEXT_ONLY_PREFIXES):
        return ParamGroup.LANGUAGE_MODEL
    return ParamGroup.OTHER


@dataclass(frozen=True)
class TrainableSummary:
    """What a freeze policy actually did, for logging and for the run manifest."""

    policy: str
    trained_groups: tuple[str, ...]
    counts_by_group: dict[str, int]
    frozen_by_group: dict[str, int]
    n_trainable: int
    n_frozen: int

    @property
    def n_total(self) -> int:
        return self.n_trainable + self.n_frozen


def summarise_trainable(names: Iterable[str], policy: str = DEFAULT_POLICY) -> TrainableSummary:
    """Apply a freeze policy to parameter names and report what happened.

    Returns counts rather than mutating anything, so the decision can be inspected and logged
    before it is applied — and so it is testable without a 27B model.
    """
    if policy not in FREEZE_POLICIES:
        raise UnknownFreezePolicy(
            f"unknown freeze policy {policy!r}; known: {sorted(FREEZE_POLICIES)}"
        )
    trained = FREEZE_POLICIES[policy]

    counts: dict[str, int] = {}
    frozen: dict[str, int] = {}
    n_trainable = n_frozen = 0

    for name in names:
        group = classify_parameter(name)
        if group in trained:
            counts[group.value] = counts.get(group.value, 0) + 1
            n_trainable += 1
        else:
            frozen[group.value] = frozen.get(group.value, 0) + 1
            n_frozen += 1

    return TrainableSummary(
        policy=policy,
        trained_groups=tuple(sorted(g.value for g in trained)),
        counts_by_group=counts,
        frozen_by_group=frozen,
        n_trainable=n_trainable,
        n_frozen=n_frozen,
    )


def apply_freeze_policy(model, policy: str = DEFAULT_POLICY) -> TrainableSummary:
    """Set `requires_grad` on a live model according to `policy`.

    Kept separate from :func:`summarise_trainable` so the policy can be reasoned about without
    a model in memory. Raises before touching anything if the policy is unknown, so a typo
    cannot silently train the whole network.
    """
    if policy not in FREEZE_POLICIES:
        raise UnknownFreezePolicy(
            f"unknown freeze policy {policy!r}; known: {sorted(FREEZE_POLICIES)}"
        )
    trained = FREEZE_POLICIES[policy]

    for name, param in model.named_parameters():
        param.requires_grad_(classify_parameter(name) in trained)

    return summarise_trainable((n for n, _ in model.named_parameters()), policy)
