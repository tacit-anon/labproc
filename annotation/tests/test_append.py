"""End-to-end smoke test for the master-corpus append: idempotency + dedup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tacit_annotator.append import append_to_master


@pytest.fixture
def example_annotations(tmp_path: Path) -> Path:
    payload = {
        "branch": "pcr",
        "annotations": [
            {
                "video_file": "good_yt_TEST.mp4",
                "ts": 90,
                "your_label": "dry_tube_no_liquid",
                "physical_state": "dry_tube_no_liquid",
                "confidence": "high",
                "screenshot_path": "good_yt_TEST/t00090.00__dry_tube_no_liquid.jpg",
                "substance_tags": "solid",
                "action_tags": "observing_static",
                "equipment_tags": "pcr_tube,ice_bath",
            },
            {
                "video_file": "good_yt_TEST.mp4",
                "ts": 270,
                "your_label": "dry_bath_incubating",
                "physical_state": "dry_bath_incubating",
                "confidence": "high",
                "screenshot_path": "good_yt_TEST/t00270.00__dry_bath_incubating.jpg",
                "substance_tags": "liquid,hot",
                "action_tags": "heating",
                "equipment_tags": "dry_bath,microtube",
            },
        ],
        "audit": [
            {
                "video_file": "good_yt_OUT.mp4",
                "branch": "pcr",
                "decision": "NON_CATEGORY_SKIP",
                "reason": "clinical chemistry, not pcr",
                "label_proposal": "",
            }
        ],
    }
    out = tmp_path / "annotations.json"
    out.write_text(json.dumps(payload))
    return out


def test_append_first_run(example_annotations, tmp_path):
    master = tmp_path / "master.xlsx"
    stats = append_to_master(example_annotations, master)
    assert stats.appended == 2
    assert stats.skipped_duplicates == 0
    assert stats.audit_added == 1
    assert master.exists()
    csv = tmp_path / "master_pcr.csv"
    assert csv.exists()


def test_append_idempotency(example_annotations, tmp_path):
    master = tmp_path / "master.xlsx"
    append_to_master(example_annotations, master)
    stats = append_to_master(example_annotations, master)
    assert stats.appended == 0
    assert stats.skipped_duplicates == 2
    assert stats.audit_added == 0
    assert stats.audit_skipped_duplicates == 1


def test_append_per_branch_csv_regenerated(example_annotations, tmp_path):
    master = tmp_path / "master.xlsx"
    stats = append_to_master(example_annotations, master)
    assert stats.csv_counts == {"pcr": 2}
