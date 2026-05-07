"""
Assemble the labeling system prompt from SKILL.md + references.

The prompt is loaded once per labeling session and prepended to every frame request.
"""

from __future__ import annotations

import functools
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

REQUIRED_REFS = [
    "SKILL.md",
    "references/labels.md",
    "references/label-rules.md",
    "references/output-format.md",
    "references/vocabularies/substance.md",
    "references/vocabularies/action.md",
    "references/vocabularies/equipment.md",
]


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing reference: {path}")
    return path.read_text()


@functools.lru_cache(maxsize=1)
def load_system_prompt(repo_root: Path | None = None) -> str:
    """
    Concatenate the skill protocol + all reference docs into a single system prompt.
    Cached after first load.
    """
    root = repo_root or REPO_ROOT
    parts = [
        "You are an expert laboratory procedure labeler. Follow the protocol below exactly.",
        "When given a frame image, return a single JSON object — no prose around it.",
        "",
    ]
    for rel in REQUIRED_REFS:
        path = root / rel
        parts.append(f"\n\n========== {rel} ==========\n")
        parts.append(_read(path))
    return "".join(parts)


def build_user_prompt(
    branch: str,
    video_file: str,
    timestamp_seconds: int,
    additional_context: str = "",
) -> str:
    """
    Per-frame user prompt. Asks for a strict JSON response shape.
    """
    schema = """
{
  "selected": "<canonical label from references/labels.md, or null to skip>",
  "confidence": "<high | medium | low>",
  "candidates_considered": ["<label1>", "<label2>", ...],
  "eliminated_by_apparatus": "<one sentence>",
  "rejected_with_reason": {"<label>": "<why rejected>"},
  "substance_tags": "<comma-separated tags from substance vocab>",
  "action_tags": "<comma-separated tags from action vocab>",
  "equipment_tags": "<comma-separated tags from equipment vocab>",
  "skip_reason": "<only if selected is null>"
}
"""
    msg = (
        f"Frame from `{video_file}` at t={timestamp_seconds}s. Branch: `{branch}`.\n"
        f"Apply the equipment-scan + substance-state protocol. If no label fits cleanly, return `selected: null` "
        f"with a `skip_reason`. Respond with a single JSON object matching this schema:\n{schema}"
    )
    if additional_context:
        msg = f"{msg}\n\nAdditional context: {additional_context}"
    return msg


def build_triage_user_prompt(branch: str, video_file: str, frames: list[tuple[int, str]]) -> str:
    """
    Triage prompt — given multiple frames at known timestamps, produce one of three decisions.
    `frames` is a list of (timestamp_seconds, image_reference) tuples for context.
    """
    schema = """
{
  "decision": "<MATCH | TAXONOMY_GAP | NON_CATEGORY_SKIP>",
  "reason": "<one sentence>",
  "label_proposal": "<only if TAXONOMY_GAP, else empty>"
}
"""
    ts_list = ", ".join(f"t={t}s" for t, _ in frames)
    return (
        f"Triage decision for `{video_file}` (branch: `{branch}`). "
        f"You have inspected frames at {ts_list}.\n"
        f"Apply the two-part triage check (equipment match + substance-state match) across the dense sample. "
        f"Return one of three decisions:\n"
        f"  - MATCH: video belongs in this category and current taxonomy covers its content.\n"
        f"  - TAXONOMY_GAP: video belongs in this category but no current label fits a procedure shown — propose a new label.\n"
        f"  - NON_CATEGORY_SKIP: video is outside this category entirely.\n\n"
        f"Respond with a single JSON object matching this schema:\n{schema}"
    )
