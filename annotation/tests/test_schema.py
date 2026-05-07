"""Schema invariants — these contracts are what every downstream consumer relies on."""

from tacit_annotator.schema import (
    ANNOTATION_COLUMNS,
    AUDIT_COLUMNS,
    BRANCH_TO_CATEGORY,
    MANIFEST_COLUMNS,
    Annotation,
    AuditEntry,
)


def test_annotation_columns_exact_count():
    assert len(ANNOTATION_COLUMNS) == 10


def test_manifest_columns_exact_count():
    assert len(MANIFEST_COLUMNS) == 7


def test_audit_columns_exact_count():
    assert len(AUDIT_COLUMNS) == 6


def test_branch_to_category_mapping():
    assert BRANCH_TO_CATEGORY["op"] == "organic_purification"
    assert BRANCH_TO_CATEGORY["pcr"] == "pcr"
    assert BRANCH_TO_CATEGORY["wb"] == "western_blot"


def test_annotation_dedup_key():
    a = Annotation(
        branch="pcr",
        video_file="v.mp4",
        timestamp_seconds=120.0,
        your_label="dry_tube_no_liquid",
        confidence="high",
        screenshot_path="",
    )
    assert a.dedup_key == ("pcr", "v.mp4", 120.0)


def test_annotation_to_row_matches_columns():
    a = Annotation(
        branch="pcr",
        video_file="v.mp4",
        timestamp_seconds=120.0,
        your_label="dry_tube_no_liquid",
        confidence="high",
        screenshot_path="v/t00120.00__dry_tube_no_liquid.jpg",
        substance_tags="solid",
        action_tags="observing_static",
        equipment_tags="pcr_tube",
    )
    row = a.to_row()
    assert len(row) == len(ANNOTATION_COLUMNS)
    assert row[0] == "pcr"
    assert row[3] == row[5]  # physical_state mirrors your_label


def test_audit_dedup_key():
    e = AuditEntry(
        video_file="v.mp4",
        branch="pcr",
        decision="NON_CATEGORY_SKIP",
        reason="not pcr",
    )
    assert e.dedup_key == ("v.mp4", "pcr", "NON_CATEGORY_SKIP", "not pcr")
