# labproc-tacit

Evaluation utilities for the **LabProc** benchmark and the **Tacit** domain-adapted V-JEPA-2.1 video encoder, accompanying our NeurIPS 2026 Evaluations & Datasets Track submission.

The benchmark dataset lives at [`huggingface.co/datasets/Labproc/labproc`](https://huggingface.co/datasets/Labproc/labproc) and the released Tacit checkpoint at [`huggingface.co/Labproc/tacit`](https://huggingface.co/Labproc/tacit).

## What's in here

```
labproc-tacit/
├── labproc_tacit/        # Importable Python package — encoder + probes
│   ├── encoder.py        # build_encoder, load_checkpoint, encode_video
│   ├── video_io.py       # extract_frames_at_timestamp, find_video_path
│   ├── data.py           # CSV loaders + controlled vocabularies
│   └── probes/           # One module per benchmark task
│       ├── psc.py
│       ├── ccr.py
│       ├── ted.py
│       ├── ted_visual.py
│       ├── vsd.py
│       └── ccr_same_state.py
│
├── scripts/              # User-facing CLI evaluation entry points
│   ├── evaluate_psc.py
│   ├── evaluate_ccr.py
│   ├── evaluate_ted.py
│   ├── evaluate_ted_visual.py
│   ├── evaluate_vsd.py
│   └── evaluate_ccr_same_state.py
│
└── reproduce_paper/      # Research-grade scripts for paper number reproduction
    ├── README.md
    ├── adapt_v3.py       # Tacit adaptation training loop
    ├── extract_features.py
    └── sweep_*.py        # Per-epoch sweep runners for each task
```

For routine evaluation use `scripts/`. The `reproduce_paper/` directory is for users who want to reproduce the paper's per-epoch comparison tables exactly; those scripts have hard-coded paths matching the original GPU environment and may need adjustment.

## Installation

### 1. Clone this repo and install the package

```bash
git clone https://huggingface.co/Labproc/labproc-tacit
cd labproc-tacit
pip install -e .
```

### 2. Install V-JEPA-2 model definition

The encoder uses Meta's V-JEPA-2 architecture. Clone the official repo somewhere on your machine:

```bash
git clone https://github.com/facebookresearch/jepa ~/.cache/torch/hub/facebookresearch_vjepa2_main
```

If you want to clone elsewhere, set the `VJEPA2_REPO_ROOT` environment variable to point at it.

### 3. Optional: install CLIP for the TED `visual_text` variant

```bash
pip install git+https://github.com/openai/CLIP.git
pip install ftfy regex
```

## Quick start

### Acquire source videos

The LabProc dataset releases annotations and a YouTube URL manifest, but does not redistribute video files. Download the videos referenced in `manifests/source_videos.csv`:

```bash
python -c "
import pandas as pd
m = pd.read_csv('manifests/source_videos.csv')
for url in m['upstream_url']:
    print(url)
" | xargs -I{} yt-dlp -o '/your/video/dir/%(id)s.%(ext)s' {}
```

### Run any benchmark task

The default checkpoint is the released Tacit at `Labproc/tacit`. Override with `--checkpoint <path>` to evaluate the unadapted V-JEPA-2.1 base or any other checkpoint.

```bash
# PSC: physical state classification
python scripts/evaluate_psc.py \
    --psc-csv annotations/op_master.csv \
    --video-root /your/video/dir \
    --checkpoint /path/to/tacit.pth \
    --checkpoint-key model_state \
    --cache /tmp/features_psc_tacit.pt

# CCR: standard pairwise ordering (reuse the PSC feature cache)
python scripts/evaluate_ccr.py \
    --psc-csv annotations/op_master.csv \
    --video-root /your/video/dir \
    --checkpoint /path/to/tacit.pth \
    --checkpoint-key model_state \
    --cache /tmp/features_psc_tacit.pt

# TED: 4-MCQ transition error detection
python scripts/evaluate_ted.py \
    --psc-csv annotations/op_master.csv \
    --ted-csv benchmark_items/ted.csv \
    --video-root /your/video/dir \
    --checkpoint /path/to/tacit.pth \
    --checkpoint-key model_state \
    --variant both

# TED-Visual: motion triplet anchor matching
python scripts/evaluate_ted_visual.py \
    --triplets-csv benchmark_items/ted_visual.csv \
    --video-root /your/video/dir \
    --checkpoint /path/to/tacit.pth \
    --checkpoint-key model_state

# VSD: visual state discrimination
python scripts/evaluate_vsd.py \
    --psc-csv annotations/op_master.csv \
    --video-root /your/video/dir \
    --checkpoint /path/to/tacit.pth \
    --checkpoint-key model_state \
    --cache /tmp/features_psc_tacit.pt
```

The PSC, CCR, and VSD scripts share a feature cache. If you pass `--cache /tmp/features_psc_tacit.pt` to all three, the encoder runs only once.

## Headline numbers

To reproduce the paper's table, run each script with `--checkpoint Labproc/tacit` and again with the base V-JEPA-2.1 checkpoint:

| Task | Tacit | Base V-JEPA-2.1 |
|------|-------|------------------|
| PSC | 31.2% | 16.2% |
| TED | 76.1% | 75.3% |
| CCR pair-acc | 58.7% | 43.9% |
| VSD aggregate | 57.8% | 50.2% |
| VSD pure-motion | 54.7% | 50.0% |
| TED-Visual Hard | 69.6% | 60.9% |
| TED-Visual Strict Hard | 66.7% | 60.6% |

Tacit checkpoint = epoch 4 of the adaptation run, loss 0.70.

## Citing this work

```bibtex
@inproceedings{labproc2026,
  title     = {LabProc and Tacit: A Benchmark and Domain-Adapted Video Encoder for Laboratory Procedure Understanding},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 Evaluations and Datasets Track},
  year      = {2026}
}
```

The author and affiliation will be filled in for the camera-ready release.

## License

CC BY 4.0 — see [LICENSE](LICENSE).

## Notes on the Same-State CCR script

`scripts/evaluate_ccr_same_state.py` is provided for users investigating alternative adaptation strategies, but **the released Tacit checkpoint is not the appropriate evaluation target for this task**. The adaptation pipeline attenuates within-state temporal coherence; see companion paper Section 6 for full details. Same-State CCR is not part of the v1 benchmark.
