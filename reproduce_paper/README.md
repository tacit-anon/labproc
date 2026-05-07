# Paper reproduction scripts

This directory contains the **research-grade scripts** used to produce the numbers in the LabProc + Tacit paper. They are preserved as-is from the original GPU environment for transparency and exact reproducibility.

For routine evaluation, use `../scripts/` instead. Those user-facing scripts call the same probe code via the cleaner `labproc_tacit` package and have sensible defaults / argparse flags. The scripts here are kept in their original form because they encode the exact training-time choices (epoch sweeps, hyperparameter explorations, dataset construction passes) that produced the headline numbers.

## What's in here

### Adaptation training

- `adapt_v3.py` — Tacit domain-adaptive continued pretraining of V-JEPA-2.1. Implements EMA target encoder + motion-weighted masking. Requires `dataset.py` (defines `LabProcDataset`) which is **not yet uploaded**; see note below.

### Feature extraction

- `extract_features.py` — Extract frozen V-JEPA-2.1 features from all annotated clips for both base and adapted conditions. Used by all the train_*.py scripts. Also requires `dataset.py`.

### Per-task probe training

- `train_psc_ccr_probes.py` — Single-checkpoint PSC + CCR runner (the source of canonical helper functions for many other scripts).
- `train_ted_probes.py` — Single-checkpoint TED runner with both probe variants.
- `train_ted_visual_siamese_probe.py` — Single-checkpoint TED-Visual siamese probe runner.
- `train_vsd_pad_probes.py` — Single-checkpoint VSD + PAD runner.
- `train_ccr_same_state.py` — Single-checkpoint Same-State CCR runner.

### Per-epoch sweeps

The following loop the corresponding train_*.py runner over base + 7 adaptation epochs to produce the per-epoch comparison tables in the paper:

- `sweep_psc_ccr_per_epoch.py` — PSC + CCR sweep
- `sweep_ccr_only_per_epoch.py` — CCR-only sweep (independent state)
- `sweep_ccr_antisymmetric_per_epoch.py` — Antisymmetric CCR variant
- `sweep_ccr_same_state_per_epoch.py` — Same-State CCR sweep (concat probe)
- `sweep_ccr_same_state_antisymmetric_per_epoch.py` — Same-State CCR antisymmetric variant
- `sweep_ted_per_epoch.py` — TED sweep
- `sweep_vsd_ted_visual_per_epoch.py` — VSD + TED-Visual sweep

## Hard-coded paths

These scripts were written to run inside a specific GPU environment and contain hard-coded paths:

- `/home/tacit_experiment/` — main working directory
- `/home/tacit_experiment/adapted_v3_ema_motion_checkpoints/` — per-epoch checkpoint directory
- `/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt` — base V-JEPA-2.1
- `/home/tacit_annotation/` — annotation CSVs
- `/home/downloads/` — source video files
- `/home/.cache/torch/hub/facebookresearch_vjepa2_main` — V-JEPA-2 repo

If you want to reproduce paper numbers exactly, you'll need to either replicate this layout or globally find-replace the paths.

## Missing: dataset.py

`adapt_v3.py` and `extract_features.py` depend on a `LabProcDataset` class defined in `dataset.py`, which has not yet been uploaded. Without this file, those two scripts cannot run; the rest can run independently because they don't use the dataset class (they pull frames directly from videos via OpenCV).

Once `dataset.py` is uploaded, place it alongside these scripts.

## Why a separate directory?

The user-facing `scripts/` and `labproc_tacit/` package are written to be portable, well-documented, and free of hard-coded paths. Refactoring `adapt_v3.py` and the `sweep_*.py` runners to that standard would have required restructuring the per-epoch comparison logic in ways that risk breaking the exact numerical reproduction. We opted to preserve them here verbatim.
