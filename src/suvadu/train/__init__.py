"""Training-side components.

Split deliberately from `suvadu.cli.train`: the logic that can be tested on a laptop
(which parameters are trainable, what a progress line says, whether a resume is legal) lives
here and is unit-tested, while the parts that genuinely need two GB10 nodes stay in the CLI.
Anything that cannot be tested without the cluster is a place bugs hide until an expensive run.
"""

from suvadu.train.freeze import (
    FREEZE_POLICIES,
    ParamGroup,
    classify_parameter,
    summarise_trainable,
)
from suvadu.train.progress import ProgressReporter
from suvadu.train.resume import ResumeState, ResumeMismatch

__all__ = [
    "FREEZE_POLICIES",
    "ParamGroup",
    "ProgressReporter",
    "ResumeMismatch",
    "ResumeState",
    "classify_parameter",
    "summarise_trainable",
]
