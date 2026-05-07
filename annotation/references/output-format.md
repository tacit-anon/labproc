# Output format

The bundle produced by this skill is consumed directly by the Tacit annotation tool's Import feature. Match the format exactly.

## Bundle layout

```
{output_root}/
├── {video_basename_1}/
│   ├── t00010.50__liquid_mixed_clear.jpg
│   ├── t00045.00__gel_running.jpg
│   └── t00120.00__gel_complete_bands_visible.jpg
├── {video_basename_2}/
│   ├── t00030.00__mixture_dissolved_hot.jpg
│   └── t00210.00__crystals_forming.jpg
└── tacit_annotations_{YYYY-MM-DD}.xlsx
```

The `{video_basename}` is the video's filename without extension, with non-alphanumeric characters replaced by underscores. Example: `good_yt_-8JVGmyiFXo.mp4` → `good_yt_-8JVGmyiFXo`.

## Filename convention

`t{HHHHH.SS}__{label}.jpg`

- `HHHHH` — integer seconds, zero-padded to 5 digits (sortable)
- `SS` — 2-decimal precision after the period
- `__` — double underscore separator
- `{label}` — canonical snake_case value from `references/labels.md` (no human-readable substitutions)

Examples:
- `t00010.50__liquid_mixed_clear.jpg` — 10.5s, PCR liquid mixed clear
- `t00150.00__crystals_forming.jpg` — 150s, OP crystals forming
- `t01020.50__bands_visible.jpg` — 17:00.5, WB bands visible

The 5-digit padding handles videos up to 99,999s (~27 hours) — far more than needed.

## Spreadsheet structure

### Sheet 1: `annotations` (required, primary)

10 columns from skill v8 onward (was 7):

| column | type | example |
|---|---|---|
| `branch` | string | `op` / `pcr` / `wb` |
| `video_file` | string | `good_yt_-8JVGmyiFXo.mp4` |
| `timestamp_seconds` | number | `150.50` |
| `physical_state` | string | `crystals_forming` (mirrors `your_label`) |
| `confidence` | string | `high` / `medium` / `low` |
| `your_label` | string | `crystals_forming` |
| `screenshot_path` | string | `good_yt_-8JVGmyiFXo/t00150.00__crystals_forming.jpg` |
| `substance_tags` | string (CSV) | `liquid,suspension,cool,cloudy,white` |
| `action_tags` | string (CSV) | `cooling_ice_bath,observing_static` |
| `equipment_tags` | string (CSV) | `erlenmeyer_flask,ice_bath,beaker` |

Notes:
- `physical_state` and `your_label` carry the same value (legacy duplicate kept for schema stability with the live tool)
- `screenshot_path` is the relative path inside the bundle root — matches the on-disk layout
- The 3 tag columns are comma-separated values from controlled vocabularies in `references/vocabularies/`. Empty strings are valid for legacy pre-v8 rows; new annotations should populate at least one tag from each of substance/action/equipment.

### Sheet 2: `video_manifest` (optional, helpful for batch overview)

7 columns:

| column | type |
|---|---|
| `video_file` | string |
| `category` | `organic_purification` / `pcr` / `western_blot` |
| `branch` | `op` / `pcr` / `wb` |
| `status` | `pending` / `in_progress` / `done` (use `done` for fully-labeled videos) |
| `annotation_count` | number |
| `folder` | string (source folder, e.g., `Videos/Organic Purification`) |
| `last_modified` | ISO 8601 timestamp |

The Tacit tool reads only `annotations` on import; `video_manifest` is for human review.

## Branch ↔ category mapping

| friendly category | branch code |
|---|---|
| `organic_purification` | `op` |
| `pcr` | `pcr` |
| `western_blot` | `wb` |

Always emit lowercase branch codes. The category column uses underscored lowercase (`organic_purification`, not `Organic Purification`).
