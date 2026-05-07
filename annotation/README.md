# Tacit Video Annotator

> End-to-end pipeline for hierarchical physical-state labeling of laboratory procedure videos.
>
> Input: raw YouTube-sourced procedure videos (PCR, organic purification, Western blot).
> Output: hierarchically-labeled training corpus, ready to import into the Tacit annotation tool or feed a vision-language model.

[![CI](https://github.com/tacit-ai/tacit-video-annotator/actions/workflows/ci.yml/badge.svg)](https://github.com/tacit-ai/tacit-video-annotator/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Skill version](https://img.shields.io/badge/skill-v9.0-green.svg)](SKILL.md)

## What it does

Converts raw lab-procedure video → hierarchically-labeled training corpus, with explicit anti-anchoring audit trail and empirically-calibrated yield priors.

## Pipeline (5 stages)

```
   ┌──────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌────────┐
   │  raw     │──▶│ triage  │──▶│ dense   │──▶│ 4-D      │──▶│ bundle │
   │  video   │   │ 10-frame│   │ extract │   │ labeling │   │ +master│
   └──────────┘   └─────────┘   └─────────┘   └──────────┘   └────────┘
                       │              │             │              │
                       ▼              ▼             ▼              ▼
                  triage_report  ffmpeg -ss    Anthropic API   audit JSON
                  (3 outcomes)   30s default   (or pluggable)  + xlsx
```

## Quickstart

```bash
# 1. install
pip install -e .

# 2. set your API key (Anthropic Claude with vision)
export ANTHROPIC_API_KEY=sk-...

# 3. run end-to-end on one video
tacit-annotate run path/to/video.mp4 \
    --branch pcr \
    --output-dir ./out \
    --master-xlsx ./master.xlsx

# Or run the whole folder
tacit-annotate batch path/to/Videos/PCR/ \
    --branch pcr \
    --output-dir ./out \
    --master-xlsx ./master.xlsx \
    --quality-tier good_yt
```

## CLI

```
tacit-annotate triage <video> --branch pcr           Run dense-sampling triage (10 frames + decision)
tacit-annotate extract <video> <out>                 Extract frames at 30s intervals
tacit-annotate label <frames_dir> --branch pcr ...   Label every frame in a directory
tacit-annotate build <annotations.json> <out>        Compile a Tacit-import-ready bundle
tacit-annotate append <input> <master.xlsx>          Idempotent append to master corpus
tacit-annotate run <video> --branch pcr              Full end-to-end pipeline (single video)
tacit-annotate batch <dir> --branch pcr              Full end-to-end pipeline (folder of videos)
```

Every subcommand has `--help`. The `run` and `batch` commands are the primary entry points.

## Programmatic API

```python
from pathlib import Path
from tacit_annotator import label, extract, bundle, append
from tacit_annotator.label import AnthropicClient

client = AnthropicClient(model="claude-sonnet-4-6")

# Triage
triage_frames = extract.extract_triage_frames(Path("video.mp4"), Path("./triage"), n=10)
decision = label.triage_video("video.mp4", "pcr", triage_frames, client)

# Dense extract + label
extract.extract_dense_frames(Path("video.mp4"), Path("./frames"), interval=30)
annotations, audits = label.label_frames_dir(Path("./frames"), "pcr", "video.mp4", client)

# Build + append
xlsx = bundle.build_bundle(annotations, frames_root=Path("./frames").parent, output_root=Path("./bundle"))
stats = append.append_to_master(Path("./annotations.json"), Path("./master.xlsx"))
```

## Architecture

```
src/tacit_annotator/
├── __init__.py          # public API
├── cli.py               # click-based CLI
├── schema.py            # dataclasses: Annotation, AuditEntry, TriageDecision
├── extract.py           # ffmpeg -ss frame extraction (NEVER fps= filter)
├── prompts.py           # SKILL.md + references → system prompt
├── label.py             # Anthropic API labeling backend (pluggable)
├── audit.py             # anti-anchoring audit JSON persistence
├── bundle.py            # annotations + frames → Tacit-import bundle
├── append.py            # idempotent master-corpus append
└── run.py               # end-to-end orchestration
```

The `LabelingClient` Protocol in `label.py` lets you swap providers (OpenAI Vision, Gemini, local VLM) without touching the rest of the pipeline.

## Repository layout

```
tacit-video-annotator/
├── SKILL.md                       # v9 protocol — read this first
├── README.md                      # you are here
├── PAPER.md                       # research-paper methodology
├── CHANGELOG.md                   # v8 → v9 diff
├── LICENSE                        # Apache 2.0
├── pyproject.toml                 # Python packaging
├── requirements.txt               # for non-poetry users
├── Makefile                       # install / test / lint / format
├── .github/workflows/ci.yml       # GitHub Actions CI
├── src/tacit_annotator/           # the package (see above)
├── tests/
│   ├── test_schema.py
│   ├── test_prompts.py
│   ├── test_extract.py
│   └── test_append.py
├── references/
│   ├── labels.md                  # 58 canonical labels (OP / PCR / WB)
│   ├── label-rules.md             # disambiguation + procedural ordering
│   ├── output-format.md           # bundle layout + xlsx schema
│   └── vocabularies/
│       ├── substance.md           # ~40 substance tags
│       ├── action.md              # ~25 action tags
│       └── equipment.md           # ~50 equipment tags
├── scripts/
│   ├── extract_frames.sh          # legacy shell wrapper
│   ├── build_bundle.py            # legacy script (use the package instead)
│   ├── append_to_master.py        # legacy script (use the package instead)
│   └── run_pipeline.sh            # convenience shell wrapper for `run`
└── examples/
    └── annotations.example.json   # example annotation input
```

## Headline numbers

| metric | value | source |
|---|---:|---|
| canonical labels (primary taxonomy) | 58 | `references/labels.md` |
| controlled-vocab tags (substance / action / equipment) | ~115 | `references/vocabularies/` |
| dimensions per annotation | 4 | v8+ schema |
| bench-video yield (30s sampling) | 30–45% | empirical, n>40 |
| RT-PCR / sample-prep yield | 50–55% | empirical, n>10 |
| triage false-skip rate (single-frame) | ~75% | measured pre-v8 |
| triage false-skip rate (dense, 10-frame) | <5% | measured v8+ |

## Design principles

- **Skip is a valid output.** 20–40% intentional skip on bench videos is target. <15% suggests over-labeling; >55% suggests taxonomy gap.
- **Anti-anchoring audit trail.** Per-frame sidecar JSON records *rejected* labels with reasons — makes anchoring bias visible to reviewers.
- **Equipment-scan elimination.** Frame's apparatus eliminates whole label groups before substance-state picks among survivors.
- **Cross-category label reuse (bidirectional).** `branch` reflects source folder; labels are reusable across branches.
- **Dedup-on-triple invariant.** `(branch, video_file, timestamp_seconds)` is the corpus-integrity key.
- **Pluggable labeling backend.** Default Anthropic API; swap to any vision-LLM by implementing `LabelingClient`.

## Categories supported

| category | branch code | label count | typical workflows |
|---|---|---:|---|
| Organic Purification | `op` | 24 | recrystallization, TLC, column chromatography, distillation, reflux, rotovap |
| PCR | `pcr` | 14 | RT-PCR, gel electrophoresis, extraction columns, thermocycling |
| Western Blot | `wb` | 19 | gel running, transfer, blocking, antibody incubation, ECL detection |

Adding a new category: extend `references/labels.md` with a new section, add disambiguation rules to `references/label-rules.md`, update `references/output-format.md`'s branch table.

## Development

```bash
make dev          # install with dev dependencies
make test         # run pytest
make lint         # ruff lint
make format       # ruff format
make typecheck    # mypy
```

CI runs on Python 3.10 / 3.11 / 3.12 with lint, typecheck, tests, and a build artifact.

## Citation

If you use this pipeline in research, please cite:

```bibtex
@software{tacit_video_annotator_2026,
  author       = {Tacit team},
  title        = {Tacit Video Annotator: Hierarchical Physical-State Labeling for Laboratory Procedure Videos},
  year         = {2026},
  version      = {9.0.0},
  url          = {https://github.com/tacit-ai/tacit-video-annotator}
}
```

See [PAPER.md](PAPER.md) for the methodology suitable for paper-form publication.

## Contributing

Open issues with: (a) videos where the pipeline mis-classifies, (b) taxonomy gaps with proposed labels, (c) yield deviations from documented priors.

PRs welcome for: new category support, additional disambiguation rules, alternative labeling backends, alternative export formats.

## License

Apache 2.0 — see [LICENSE](LICENSE).
