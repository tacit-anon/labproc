---
name: tacit-video-annotator
version: 9.0.0
description: Autonomous physical-state labeling for lab procedure videos with hierarchical 4-D annotations, dense-sampling triage, and calibrated yield priors. Output bundle drops directly into Tacit annotation tool's Import API.
license: Apache-2.0
---

# Tacit Video Annotator (v9)

Autonomous, audit-traceable physical-state labeling for lab procedure videos (Organic Purification, PCR, Western Blot). The output bundle (screenshots folder + xlsx) is shaped to match the Tacit annotation tool's Import feature — no manual reformatting needed.

## What's new in v9 (vs v8)

- **Empirically-calibrated yield priors** per video archetype (bench / sample-prep / educational) — used as quality-gate signals at batch time.
- **Bidirectional cross-category label reuse** explicitly authorized (was PCR→WB only; now also WB→PCR e.g. `buffer_preparation` for PCR gel-prep frames).
- **Skip-as-signal** quantified: 20–40% skip is target, <15% suggests labeler over-labeling, >55% suggests taxonomy gap or wrong category routing.
- **Master corpus integration** documented (10-col schema, dedup-on-triple invariant, `video_manifest` + `video_audit` sheets).
- **Triage-gate justification** strengthened: single-frame triage measured at ~75% false-skip rate; dense triage closes that to <5%.

See `CHANGELOG.md` for the full diff.

## When this skill applies

- User has procedure videos (Organic Purification, PCR, Western Blot) and wants autonomous labeling at frame intervals.
- User wants training-data-scale annotation across many videos in a batch.
- User asks to "annotate", "label", "process", or "tag" video frames against a known taxonomy.

## High-level workflow

For each video the user wants annotated:

1. **Inspect** — `ffprobe` to confirm duration, resolution, codec.
2. **Triage at dense sample** — extract 10 evenly-spaced frames, two-part check (equipment scan + substance-state scan), decide MATCH / TAXONOMY-GAP / NON-CATEGORY-SKIP.
3. **Extract dense frames** at 30s intervals using `scripts/extract_frames.sh` (uses `-ss`; never the `fps=` filter).
4. **View each frame** with vision capabilities.
5. **Match to the category's label set** — `references/labels.md` has the canonical 58-label taxonomy with descriptions.
6. **Apply 4-D labels** — primary label + substance_tags + action_tags + equipment_tags from controlled vocabularies.
7. **Apply disambiguation rules** — `references/label-rules.md` before labeling ambiguous frames.
8. **Skip frames without a clean match** — empty setups, transitions, redundant scenes get omitted, not force-labeled.
9. **Compile the bundle** with `scripts/build_bundle.py` — produces `{output_root}/{video_basename}/t{padded_ts}__{label}.jpg` files plus the spreadsheet.
10. **Append to master corpus** with `scripts/append_to_master.py` — enforces 10-col schema, dedupes against `(branch, video_file, ts)`, updates manifest + audit sheets.

## Triage gate — DENSE sampling required, single-frame triage forbidden

**Empirical observation:** single-frame triage has a measured ~75% false-skip rate on edge-case videos. Failure modes are systematic:
- Triage frame catches transition / setup / equipment-only shot, missing the actual procedure.
- Triage frame is genuinely ambiguous (multiple sequential procedures across the timeline).
- Triage frame happens to be a title slide, presenter shot, or end card.

**Mandatory dense-sampling protocol:**

1. Extract **10 evenly-spaced frames** from each video (script samples at `i × duration / 11` for `i ∈ [1, 10]`, covering ~9% to ~91%).
2. View **at least 3–5** of those frames before assigning a validity vote OR a label, with at least one frame each from early third (frames 1–3), middle third (4–7), late third (8–10).
3. Apply equipment-scan + substance-state checks to *each* viewed frame. Video classification is a majority vote across the dense sample, not a single-frame guess.

Single-frame triage is permitted only as a "preview" — it can never be the basis for a final classification or skip decision.

### Two-part triage check (both must pass)

1. **Equipment match** — does the apparatus visible in the frame correspond to procedures the label set covers?

2. **Substance-state match** — is there actual procedural content matching the labels? Equipment alone is not enough; you need the procedural state visible.

If either check fails, mark the video for manual review. **Distinguish two cases:**

- **Procedure is genuinely outside the category** → SKIP. Legitimate reject. Log in `video_audit` sheet as `NON_CATEGORY_SKIP`.
- **Procedure IS in the category but no current label fits** → DO NOT SKIP. Log it as a TAXONOMY GAP with a label proposal. The right response to taxonomy gaps is to expand the taxonomy, not skip the data.

After triage, produce a `triage_report.md` listing each video, its decision, and the reasoning. TAXONOMY-GAP entries should include a label proposal.

## Anchoring guard — evaluate against the FULL label taxonomy

When humans review your output, they see only the labels you chose, not the ones you rejected. This makes anchoring bias invisible. To prevent this:

**Two-pass check per frame:**

1. **Equipment scan (eliminates by absence):** what apparatus is visible? Anything NOT visible eliminates whole groups of labels. State the eliminating cue once per video and apply silently per frame.

2. **Substance/state scan (picks among survivors):** of the remaining candidates, which has its description visibly demonstrated? If two compete, `references/label-rules.md` breaks the tie. If none clearly wins, skip the frame.

**Emit an audit trail.** Per-frame sidecar JSON:

```json
{
  "frame": "t00210.jpg",
  "timestamp_seconds": 210,
  "candidates_considered": ["mixture_crude_unreacted", "mixture_dissolved_hot", "crystals_forming", "crystals_complete"],
  "eliminated_by_apparatus": "no TLC plate, no chromatography column → labels 5-14 eliminated",
  "selected": "crystals_forming",
  "rejected_with_reason": {
    "mixture_crude_unreacted": "crystals visible, not opaque pre-heat crude",
    "mixture_dissolved_hot": "ice bath visible, cooled not hot",
    "crystals_complete": "crystals still in mother liquor, no Buchner filter"
  },
  "confidence": "high"
}
```

Audit file lives alongside the bundle as `audit_{video_basename}.json`.

## Hierarchical labeling — REQUIRED (4-D annotation)

Single-label-per-frame collapses compositional structure that the downstream world model needs. Every annotation row carries **four label fields**:

| field | source | example |
|---|---|---|
| `your_label` (primary) | canonical 58-label taxonomy (`references/labels.md`) | `crystals_forming` |
| `substance_tags` | `references/vocabularies/substance.md` | `liquid,multi_phase,settled,brown,yellow` |
| `action_tags` | `references/vocabularies/action.md` | `pouring,filtering_gravity` |
| `equipment_tags` | `references/vocabularies/equipment.md` | `fluted_funnel,erlenmeyer_flask,ring_stand,hot_plate` |

Tags are comma-separated within a single column. **Apply at least one tag from each of substance/action/equipment per frame.** Empty tag fields are not acceptable for new annotations — if no action is happening, use `observing_static`.

Read the three vocabulary files before starting any new labeling work. They are the controlled vocabularies — don't invent tags outside them. If a procedure genuinely lacks a tag, propose the addition (same flow as a `your_label` taxonomy gap).

## Cross-category label reuse (v9 — bidirectional)

Some labels apply across categories. v9 explicitly authorizes reuse in either direction:

- `tube_in_vortex`, `tube_in_microcentrifuge`, `dry_bath_incubating` (defined under PCR) — apply to OP and WB sample-prep frames.
- `buffer_preparation`, `protein_sample_with_buffer` (defined under WB) — apply to PCR gel-prep and reagent-prep frames.
- `solvent_dispensing`, `analytical_weighing` (defined under OP) — apply to PCR/WB reagent-handling frames.

When a frame from one branch shows a procedure that has a clear label in another branch, use that label. The `branch` column always reflects the **source category** (where the video is filed), but labels are reusable across the corpus.

## Critical rules

### Frame extraction (functional — don't get this wrong)

Always use `ffmpeg -ss <exact_seconds> -i <video> -frames:v 1 -q:v 2 -loglevel error <output>` to grab one frame at a precise timestamp. The `-vf "fps=1/30"` filter pattern *drifts* — frames don't end up at the timestamps you expect, which silently corrupts the dataset. Use `scripts/extract_frames.sh`.

Don't apply `scale=` filters. Source videos are often 480×360 (YouTube). Upscaling adds interpolation artifacts that mislead analysis. Work at native resolution.

### Labeling judgment

- **Match descriptively, not vaguely.** Each label has a one-line description. The frame must visibly demonstrate that description.
- **Skip is a valid annotation outcome.** If no label matches cleanly, leave the frame out — don't force-fit.
- **Use the confidence field honestly.** `high` = unambiguous match. `medium` = likely match but visual ambiguity exists. `low` = procedural inference rather than direct visual match.
- **Apply procedural ordering.** Some labels imply temporal sequence — e.g., crystals can't be `complete` before they've been collected onto a filter. See `references/label-rules.md`.

### Output format (contract)

Output is consumed by the Tacit annotation tool's Import feature. Match this exactly:

- Filenames: `t{HHHHH.SS}__{label}.jpg` (5-digit zero-padded seconds, 2-decimal precision, double underscore, snake_case label).
- Folder layout: `{output_root}/{video_basename}/{filename}.jpg`.
- Spreadsheet: 10 columns — `branch | video_file | timestamp_seconds | physical_state | confidence | your_label | screenshot_path | substance_tags | action_tags | equipment_tags`.
- Optional sheets: `video_manifest`, `video_audit`.

Full schema and examples in `references/output-format.md`.

## Empirically-calibrated yield priors (v9)

Use these as batch-time quality-gate signals. If observed yield deviates significantly, investigate before shipping.

| video archetype | expected labeled-frame yield | rows / video (30s sampling, 5–10 min) | notes |
|---|---:|---:|---|
| Bench procedure (full workflow) | 30–45% | 5–8 | OP recrystallization, WB transfer, etc. |
| RT-PCR / sample prep (dense procedural) | 50–55% | 8–10 | tubes + reagents + thermocycler dense-shot |
| Educational / talking-head + bench cutaways | ~10% | 1–3 | university tutorials, brand explainers |
| Promo / unboxing | ~10% | 1–2 | product showcase, equipment overview |
| Out-of-category mis-categorization | 0% | 0 | wrong-folder video, log to `video_audit` |

**Anomaly thresholds:**
- Bench video at <15% yield → check for over-skipping or undocumented taxonomy gap.
- Bench video at >55% yield → check for force-labeling redundant or transition frames.
- Educational at >25% yield → likely valid, but spot-check for over-labeling presenter shots.

## Default parameters

- **Sampling interval:** 30 seconds. For short or fast-moving videos, ask the user about denser sampling (10–15s).
- **Confidence floor:** include all confidences by default. If user wants a clean dataset, offer to drop `low` confidence rows on export.
- **Skip ratio expectation:** 20–40% on bench videos at 30s sampling — sign of honest labeling.

## Scripts

- `scripts/extract_frames.sh <video> <output_dir> [interval_seconds=30]` — extracts frames at exact timestamps using `-ss`. Output filenames are `t{seconds_padded5}.jpg`.
- `scripts/build_bundle.py <annotations.json> <output_root>` — compile the final bundle (rename frames with labels, organize into per-video folders, generate xlsx).
- `scripts/append_to_master.py <annotations.json> <master.xlsx>` — append new rows to the master corpus xlsx, dedupe against `(branch, video_file, ts)`, update `video_manifest` + `video_audit` sheets, regenerate per-branch CSV exports.

## Reasonable per-batch flow

1. Confirm scope with the user: which folder, which category, which videos, what interval?
2. Per video: triage → extract → label → save annotation JSON.
3. After all videos labeled, run `build_bundle.py` once for the bundle, then `append_to_master.py` for corpus integration.
4. Tell the user where the bundle is and how to import it into the Tacit tool.

If labeling 10+ videos in a session, work video-by-video and save partial progress so a long-running session can recover from interruptions. Build and append are idempotent (dedup on triple).

## Reference files

- `references/labels.md` — canonical 58-label catalog per category (OP / PCR / WB) with descriptions.
- `references/label-rules.md` — disambiguation rules and procedural ordering for ambiguous cases.
- `references/output-format.md` — exact bundle layout, spreadsheet schema, filename convention.
- `references/vocabularies/substance.md` — controlled vocabulary for `substance_tags`.
- `references/vocabularies/action.md` — controlled vocabulary for `action_tags`.
- `references/vocabularies/equipment.md` — controlled vocabulary for `equipment_tags`.

Read these before labeling. Especially `label-rules.md` — most mistakes come from wrong rule application, not wrong vision.
