"""Prompt assembly tests — ensure every reference is loaded and the schema mention is in the user prompt."""

from tacit_annotator.prompts import build_triage_user_prompt, build_user_prompt, load_system_prompt


def test_system_prompt_includes_all_references():
    sp = load_system_prompt()
    assert "labels.md" in sp
    assert "label-rules.md" in sp
    assert "output-format.md" in sp
    assert "substance.md" in sp
    assert "action.md" in sp
    assert "equipment.md" in sp


def test_user_prompt_requests_json_shape():
    up = build_user_prompt("pcr", "v.mp4", 120)
    assert "selected" in up
    assert "candidates_considered" in up
    assert "JSON" in up


def test_triage_prompt_lists_three_decisions():
    tp = build_triage_user_prompt("pcr", "v.mp4", [(50, "f1"), (200, "f2")])
    assert "MATCH" in tp
    assert "TAXONOMY_GAP" in tp
    assert "NON_CATEGORY_SKIP" in tp
