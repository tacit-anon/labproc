# Tacit Video Annotator: A Methodology for Hierarchical Physical-State Labeling of Laboratory Procedure Videos

> Methodology brief suitable for paper-form publication. Pairs with the production pipeline in this repository.

## Abstract

We present a 6-stage pipeline for converting raw, web-sourced laboratory procedure videos (PCR, organic purification, Western blot) into a hierarchically-labeled training corpus for physical-AI world models. The pipeline introduces three methodological contributions: (i) **dense-sampling triage**, which empirically reduces the false-skip rate of single-frame video classification from ~75% to <5%; (ii) **4-dimensional hierarchical annotation** (primary state + substance + action + equipment) that captures compositional structure absent from single-label datasets at marginal labor cost; and (iii) **per-frame anti-anchoring audit trails** that make labeling bias visible to reviewers. We report empirically-calibrated yield priors per video archetype that serve as batch-time quality-gate signals. The pipeline is open-source, category-agnostic, and produces output that drops directly into downstream annotation tools.

## 1. Problem statement

Physical-AI world models — particularly category-specific V-JEPA enrichment for laboratory procedure understanding — require dense, hierarchically-labeled procedure data covering real-world physical fidelity. Existing options have well-known limitations:

- **Public lab-procedure datasets** are coarse (one label per video) and small (<1,000 hrs).
- **Synthetic data** is expensive and fails to capture the variance of glassware staining, lighting, hand occlusion, and procedural deviation present in real benches.
- **Manual annotation at scale** is cost-prohibitive (>$10/frame for expert labelers) and produces inter-annotator agreement issues.

The pipeline described here addresses these gaps for a specific niche: web-sourced procedure videos, hierarchical labels, automated single-labeler annotation with audit traceability.

## 2. Pipeline architecture

### 2.1 Stage 1 — Source curation and quality gating

Videos are pre-classified at ingestion into 3 quality tiers (`good_yt` / `ok_yt` / `bad_yt`) by a human curator using a 30-second skim. Only `good_yt` enters annotation. Empirical observation: lower tiers yield <5% labelable frames, making expected reward per labor unit negative.

### 2.2 Stage 2 — Dense triage (validity filter)

For each video, we extract 10 evenly-spaced frames at timestamps `t_i = i × duration / 11` for `i ∈ {1, ..., 10}`, covering 9% to 91% of the timeline.

For each frame, a two-part check is applied:

1. **Equipment scan.** Does the visible apparatus correspond to procedures the category's label set covers?
2. **Substance-state scan.** Is procedural content visible (versus setup-only, transition, or presenter shot)?

The video receives one of three classifications via majority vote across the dense sample:

- `MATCH` — proceed to dense extraction.
- `TAXONOMY_GAP` — procedure IS in category but no current label fits; log a label proposal and proceed.
- `NON_CATEGORY_SKIP` — procedure outside category; log to `video_audit` sheet and drop.

**Empirical justification.** We measured the false-skip rate of single-frame triage at ~75% on edge-case videos (n=23 from a manual-review set). The systematic failure modes are:
- Title slide / intro card lands at the triage timestamp.
- Equipment-only "B-roll" frame misses the procedural content elsewhere in the timeline.
- Mid-video transition between two procedures masks the dominant content.

Dense 10-frame triage closes the false-skip rate to <5% (n=23 same set, re-evaluated) at ~10× the compute cost — a favorable trade-off when downstream annotation cost is the dominant term.

### 2.3 Stage 3 — Dense frame extraction

Frames are extracted at user-specified intervals (default: 30s) using `ffmpeg -ss <timestamp> -i <video> -frames:v 1 -q:v 2`. We deliberately avoid the `-vf "fps=1/30"` filter, which drifts: emitted frames do not occupy the requested timestamps, silently corrupting downstream alignment.

We do not apply `scale=` filters. Source videos are typically 480×360 (YouTube). Upscaling introduces interpolation artifacts that mislead vision models without adding genuine detail. Native resolution is preserved.

### 2.4 Stage 4 — Hierarchical 4-D labeling

Each labeled frame carries four orthogonal label dimensions:

| dimension | source | example |
|---|---|---|
| `your_label` (primary) | 58-label canonical taxonomy | `crystals_forming` |
| `substance_tags` | substance vocabulary (~40 tags) | `liquid,multi_phase,cool,white` |
| `action_tags` | action vocabulary (~25 tags) | `cooling_ice_bath,observing_static` |
| `equipment_tags` | equipment vocabulary (~50 tags) | `erlenmeyer_flask,ice_bath,beaker` |

The labeling protocol enforces an **anti-anchoring discipline**:

1. **Equipment scan (eliminates by absence).** The frame's visible apparatus eliminates whole label groups. For example, in an OP video frame with no TLC plate, all four `tlc_*` labels are eliminated; with no chromatography column, six column-related labels are eliminated. This is stated once per video and applied silently per frame.

2. **Substance/state scan (picks among survivors).** Among remaining candidates, the labeler selects the one whose description is visibly demonstrated. If two compete, disambiguation rules break the tie. If none clearly wins, the frame is **skipped**, not force-labeled.

**Skip is a valid annotation outcome.** Empty setups, transitions, title cards, presenter shots, and redundant frames are skipped. We measure 20–40% intentional skip rate on bench videos at 30s sampling as the target band; values outside this band trigger review (over-labeling vs taxonomy gap).

### 2.5 Stage 5 — Per-frame audit trail

For each labeled frame, the pipeline emits a sidecar JSON record:

```json
{
  "frame": "t00210.jpg",
  "timestamp_seconds": 210,
  "candidates_considered": ["mixture_crude_unreacted", "mixture_dissolved_hot", "crystals_forming", "crystals_complete"],
  "eliminated_by_apparatus": "no TLC plate, no chromatography column → labels 5–14 eliminated",
  "selected": "crystals_forming",
  "rejected_with_reason": {
    "mixture_crude_unreacted": "crystals visible, not opaque pre-heat crude",
    "mixture_dissolved_hot": "ice bath visible, cooled not hot",
    "crystals_complete": "crystals still in mother liquor, no Buchner filter"
  },
  "confidence": "high"
}
```

The audit trail makes anchoring bias detectable in spot-checks: a reviewer can verify in seconds that the labeler considered and rejected competing labels with stated reasoning, rather than anchoring on the first plausible label. This is a precondition for downstream model error analysis — when a model misclassifies, the audit reveals whether the training label was contested at annotation time.

### 2.6 Stage 6 — Bundle compilation and corpus integration

The pipeline produces a strict-contract output:
- **Filename:** `t{HHHHH.SS}__{label}.jpg` (5-digit zero-padded seconds, 2-decimal precision, double-underscore separator, snake_case label).
- **Folder layout:** `{output_root}/{video_basename}/{filename}.jpg`.
- **Spreadsheet:** 10 columns (`branch`, `video_file`, `timestamp_seconds`, `physical_state`, `confidence`, `your_label`, `screenshot_path`, `substance_tags`, `action_tags`, `equipment_tags`).
- **Optional sheets:** `video_manifest` (per-video status, annotation count, source folder, last-modified), `video_audit` (skip decisions and taxonomy-gap proposals).

Master corpus integration is governed by a **dedup-on-triple invariant**: `(branch, video_file, timestamp_seconds)`. Re-runs are idempotent; partial-progress re-appends preserve corpus integrity across multi-day annotation sessions.

## 3. Empirically-calibrated yield priors

Across n>40 fully-densified videos in our internal corpus (306 rows / 67 videos as of 2026-05-04), we observe the following labeled-frame yield distributions:

| video archetype | n | yield median | yield range | rows / video (5–10 min, 30s sampling) |
|---|---:|---:|---:|---:|
| Bench procedure (full workflow) | 18 | 38% | 30–45% | 5–8 |
| RT-PCR / sample prep (dense procedural) | 6 | 52% | 45–55% | 8–10 |
| Educational / talking-head + bench cutaways | 8 | 11% | 7–15% | 1–3 |
| Promo / unboxing | 4 | 9% | 5–12% | 1–2 |
| Out-of-category mis-categorization | 2 | 0% | 0% | 0 |

These priors function as batch-time quality-gate signals: a bench video yielding <15% triggers an over-skipping investigation; a bench video yielding >55% triggers a force-labeling review.

## 4. Methodological contributions

We claim three contributions defensible in a paper:

1. **Dense-sampling triage** measurably reduces false-skip rate from ~75% (single-frame) to <5% on edge-case videos. This is the most novel methodological contribution.

2. **4-D hierarchical labels** produce strictly more downstream signal than primary-only labels at marginal labor cost (~25% extra annotation time, ~3× information density). This generalizes to any procedure-video labeling task with a controlled vocabulary.

3. **Per-frame anti-anchoring audit trail** makes labeling bias visible — a property absent from single-label datasets and required for trustworthy model error analysis at scale.

## 5. Open methodological gaps

Honest in-paper limitations to flag:

- **No inter-annotator agreement measurement.** Current corpus is single-labeler. Cohen's kappa or Krippendorff's alpha measurement requires a held-out double-labeled subset.
- **Confidence calibration is heuristic.** The `high` / `medium` / `low` field has not been validated against downstream model error distributions.
- **Skip decisions not labeled.** ~25–40% of frames are dropped without metadata, losing potential negative-example signal.
- **Yield priors derived from small-n samples.** The educational-video archetype (n=8) is the weakest-supported prior; a larger sample would tighten the range estimates.

## 6. Reproducibility

The pipeline is fully open-source under Apache 2.0. The repository contains:
- The skill protocol (`SKILL.md`) — the labeling rubric.
- The label catalog and disambiguation rules (`references/`).
- Frame extraction and bundle-build scripts (`scripts/`).
- Worked examples (`examples/`).

A reproduction run requires only `ffmpeg`, Python 3.10+, and `openpyxl`. No proprietary dependencies.

## 7. Citation

```bibtex
@software{tacit_video_annotator_2026,
  author       = {Tacit team},
  title        = {Tacit Video Annotator: Hierarchical Physical-State Labeling for Laboratory Procedure Videos},
  year         = {2026},
  version      = {9.0.0},
  url          = {https://github.com/tacit-ai/tacit-video-annotator}
}
```
