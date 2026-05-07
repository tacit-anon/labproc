# Changelog

## v9.0.0 — 2026-05-05

### Added
- **Empirically-calibrated yield priors** per video archetype (`SKILL.md` §"Empirically-calibrated yield priors"). Bench: 30–45%, RT-PCR/sample-prep: 50–55%, educational: ~10%, promo: ~10%. Used as batch-time quality-gate signals.
- **Anomaly thresholds** for batch QA: bench <15% → over-skipping check, bench >55% → force-labeling check, educational >25% → presenter-shot check.
- **Bidirectional cross-category label reuse** explicitly authorized (`SKILL.md` §"Cross-category label reuse"). v8 only allowed PCR→WB; v9 allows any direction (e.g. WB `buffer_preparation` for PCR gel-prep frames, OP `solvent_dispensing` for PCR reagent handling).
- **`scripts/append_to_master.py`** — corpus-append script. Enforces 10-col schema, dedupes on `(branch, video_file, timestamp_seconds)`, updates `video_manifest` + `video_audit` sheets, regenerates per-branch CSV exports.
- **Triage-gate empirical justification** (`SKILL.md` §"Triage gate") — single-frame triage measured at ~75% false-skip; dense 10-frame triage at <5%.
- **Skip-as-signal quantification** — 20–40% target skip on bench videos with directional anomaly thresholds.

### Changed
- **`SKILL.md`** — restructured around a 10-step per-video flow (was 7-step). Added the master-corpus append step.
- **`references/output-format.md`** — minor wording clarifications; schema unchanged (10 columns).

### Unchanged from v8 (stable contracts)
- 58-label primary taxonomy.
- `substance_tags` / `action_tags` / `equipment_tags` controlled vocabularies.
- Spreadsheet schema (10 columns; sheets `annotations`, `video_manifest`, `video_audit`).
- Filename convention `t{HHHHH.SS}__{label}.jpg`.
- Frame extraction protocol (ffmpeg `-ss`, native resolution, no scaling).

### Migration from v8
- No breaking changes. v8 bundles import unchanged into v9 master corpora.
- New `append_to_master.py` is a strict superset of the inline append patterns used in v8 sessions.

---

## v8.0.0 — 2026-05-02

### Added
- **Hierarchical 4-D labels** — primary label + substance + action + equipment tags (was single-label).
- Three controlled vocabularies under `references/vocabularies/`.
- 10-column spreadsheet schema (was 7 columns).

### Changed
- Mandatory dense-sampling triage (was optional single-frame).
- Anti-anchoring audit trail mandatory (was implicit).

### Migration from v7
- v7 single-label rows kept as-is; tag columns left empty for legacy entries. New annotations must populate at least one tag from each axis.
