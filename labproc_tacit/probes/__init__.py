"""
labproc_tacit.probes — Probe architectures and evaluation runners.

One module per benchmark task. Each module exposes a `run_<task>` function
that takes a feature cache (mapping clip_id or (video, ts) to a 1024-dim
tensor) plus task-specific items, and returns a results dict.
"""

from labproc_tacit.probes.psc import PSCProbe, run_psc_groupkfold
from labproc_tacit.probes.ccr import (
    PairwiseOrderProbe,
    construct_ccr_groups,
    run_ccr_leave_one_out,
)
from labproc_tacit.probes.vsd import BinaryProbe, run_vsd

__all__ = [
    "PSCProbe", "run_psc_groupkfold",
    "PairwiseOrderProbe", "construct_ccr_groups", "run_ccr_leave_one_out",
    "BinaryProbe", "run_vsd",
]
