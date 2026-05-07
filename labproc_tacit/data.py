"""
labproc_tacit/data.py — CSV loaders for the LabProc benchmark tasks.

This module provides one loader per task, plus the controlled vocabulary
constants that the benchmark uses. All loaders return lists of dicts with
the same canonical key set (`clip_id`, `video_file`, `video_path`,
`timestamp`, ...).

The CCR and VSD tasks do not have their own CSVs in v1; they are derived
from the master PSC annotation CSV via `construct_ccr_groups()` and
`construct_vsd_pair_items()`.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from labproc_tacit.video_io import find_video_path


# =============================================================================
# Controlled vocabulary — 25 physical states for the OP branch
# =============================================================================

VALID_OP_LABELS = [
    "mixture_crude_unreacted", "mixture_dissolved_hot",
    "crystals_forming", "crystals_complete",
    "tlc_plate_dry", "tlc_plate_spotted", "tlc_running", "tlc_developed",
    "column_dry", "column_packed", "column_equilibrated",
    "sample_loaded", "fractions_collecting", "fraction_analysis",
    "lle_two_phase_settled", "lle_draining_lower_layer",
    "distillation_setup_running", "distillate_collecting", "reflux_running",
    "rotovap_running",
    "vacuum_filtration_general", "gravity_filtration_hot",
    "analytical_weighing", "solvent_dispensing", "titration_running",
]
LABEL_TO_IDX = {label: i for i, label in enumerate(VALID_OP_LABELS)}


# =============================================================================
# VSD confusion pairs — visually similar states with same textual descriptions
# =============================================================================
#
# Each entry: (label_A, label_B, why_visually_similar, what_distinguishes_them)
# Discrimination requires fine-grained visual cues that text descriptions cannot
# capture. VSD is constructed from the master PSC annotation CSV by filtering
# items whose label is one of the two members of each pair.

VSD_PAIRS = [
    (
        "lle_two_phase_settled", "lle_draining_lower_layer",
        "separatory funnel containing two visible layers",
        "operator hand on stopcock + layer level dropping",
    ),
    (
        "column_packed", "column_equilibrated",
        "glass column with white silica",
        "liquid flowing through column + dripping into receiving flask",
    ),
    (
        "mixture_dissolved_hot", "reflux_running",
        "flask on hot plate with clear/colored liquid",
        "condenser presence + boiling vapor + temperature stability",
    ),
    (
        "fractions_collecting", "fraction_analysis",
        "small vials/test tubes with collected liquid",
        "active dripping into tubes (collecting) vs static analysis under UV",
    ),
    (
        "crystals_forming", "crystals_complete",
        "flask with white solid material",
        "turbidity dynamics (forming) vs settled crystals (complete)",
    ),
    (
        "distillation_setup_running", "reflux_running",
        "flask with condenser, heat applied",
        "collection flask filling (distillation) vs liquid returning to flask (reflux)",
    ),
]


# =============================================================================
# PSC: master 4-D annotation file
# =============================================================================

def load_psc_data(psc_csv: str | Path, video_root: str | Path) -> list[dict]:
    """Load the master OP annotation CSV (e.g., op_master.csv).

    Filters out clips whose physical_state is not in VALID_OP_LABELS, and
    clips whose source video can't be located under video_root.
    """
    items = []
    with open(psc_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = find_video_path(row["video_file"], video_root)
            if video_path is None:
                continue
            if row["physical_state"] not in LABEL_TO_IDX:
                continue
            items.append({
                "clip_id": row["clip_id"],
                "video_file": row["video_file"],
                "video_path": video_path,
                "timestamp": float(row["timestamp_seconds"]),
                "raw_label": row["physical_state"],
                "label": row["physical_state"],
            })
    return items


# =============================================================================
# CCR: derived from PSC items (no separate CSV)
# =============================================================================

def construct_ccr_groups(psc_items: list[dict]) -> list[dict]:
    """Group PSC items by source video and dedupe consecutive same-state clips.

    Returns one group per video that has >=4 distinct-state clips.
    Each group has keys `video_file`, `items` (list of psc_items in temporal
    order), and `n_clips`.
    """
    by_video = defaultdict(list)
    for item in psc_items:
        by_video[item["video_file"]].append(item)

    groups = []
    for video_file, video_items in by_video.items():
        video_items.sort(key=lambda x: x["timestamp"])
        deduped = [video_items[0]]
        for item in video_items[1:]:
            if item["raw_label"] != deduped[-1]["raw_label"]:
                deduped.append(item)
        if len(deduped) >= 4:
            groups.append({
                "video_file": video_file,
                "items": deduped,
                "n_clips": len(deduped),
            })
    return groups


# =============================================================================
# TED: separate question/options CSV
# =============================================================================

def load_ted_data(ted_csv: str | Path, video_root: str | Path) -> list[dict]:
    """Load TED 4-MCQ items from a CSV with columns: ted_id, video_file,
    timestamp_seconds, question, option_a, option_b, option_c, option_d,
    correct_option."""
    items = []
    with open(ted_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = find_video_path(row["video_file"], video_root)
            if video_path is None:
                continue
            options = [row["option_a"], row["option_b"],
                       row["option_c"], row["option_d"]]
            if not all(o in LABEL_TO_IDX for o in options):
                continue
            correct_idx = {"a": 0, "b": 1, "c": 2, "d": 3}.get(
                row["correct_option"].lower())
            if correct_idx is None:
                continue
            items.append({
                "ted_id": row["ted_id"],
                "video_file": row["video_file"],
                "video_path": video_path,
                "timestamp": float(row["timestamp_seconds"]),
                "question": row["question"],
                "options": options,
                "option_indices": [LABEL_TO_IDX[o] for o in options],
                "correct_idx": correct_idx,
            })
    return items


# =============================================================================
# TED-Visual: triplet anchor / correct / distractor
# =============================================================================

def load_ted_visual_data(triplets_csv: str | Path,
                          video_root: str | Path) -> list[dict]:
    """Load TED-Visual triplets. Marks `strict_hard` when difficulty=='hard'
    AND anchor_state == correct_state == distractor_state (all three clips
    share the same nominal physical state).
    """
    items = []
    with open(triplets_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            anchor_path = find_video_path(row["anchor_video"], video_root)
            correct_path = find_video_path(row["correct_video"], video_root)
            distractor_path = find_video_path(row["distractor_video"], video_root)
            if not (anchor_path and correct_path and distractor_path):
                continue

            anchor_state = row.get("anchor_state", "") or ""
            correct_state = row.get("correct_state", "") or ""
            distractor_state = row.get("distractor_state", "") or ""
            strict_hard = (
                row["difficulty"] == "hard"
                and anchor_state and correct_state and distractor_state
                and anchor_state == correct_state == distractor_state
            )

            items.append({
                "triplet_id": row["triplet_id"],
                "difficulty": row["difficulty"],
                "anchor_video": row["anchor_video"],
                "anchor_path": anchor_path,
                "anchor_ts": float(row["anchor_ts"]),
                "correct_path": correct_path,
                "correct_ts": float(row["correct_ts"]),
                "distractor_path": distractor_path,
                "distractor_ts": float(row["distractor_ts"]),
                "anchor_state": anchor_state,
                "correct_state": correct_state,
                "distractor_state": distractor_state,
                "strict_hard": strict_hard,
            })
    return items


# =============================================================================
# Same-State CCR: groups of ≥3 clips per video sharing one physical state
# =============================================================================

def load_same_state_ccr_groups(groups_csv: str | Path,
                                video_root: str | Path) -> list[dict]:
    """Load Same-State CCR groups. CSV columns: group_id, video_file,
    state_label, ts_1, ts_2, ts_3, ts_4, ts_5 (variable; ts_4 and ts_5
    optional).
    """
    groups = []
    with open(groups_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = find_video_path(row["video_file"], video_root)
            if video_path is None:
                continue

            timestamps = []
            for j in range(1, 6):
                ts_val = row.get(f"ts_{j}")
                if ts_val and ts_val.strip() != "":
                    timestamps.append(float(ts_val))

            if len(timestamps) < 3:
                continue

            groups.append({
                "group_id": row["group_id"],
                "video_file": row["video_file"],
                "video_path": video_path,
                "state": row["state_label"],
                "timestamps": sorted(timestamps),
            })
    return groups
