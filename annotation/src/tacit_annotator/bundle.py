"""
Bundle compilation — annotations.json + raw frames -> Tacit-import-ready bundle.

Output layout:
    {output_root}/
    ├── {video_basename}/
    │   └── t{HHHHH.SS}__{label}.jpg
    └── tacit_annotations_{YYYY-MM-DD}.xlsx
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
from pathlib import Path

from openpyxl import Workbook

from .schema import ANNOTATION_COLUMNS, MANIFEST_COLUMNS, BRANCH_TO_CATEGORY, Annotation


def _basename(video_file: str) -> str:
    """Match the SKILL convention: stem with non-alnum -> underscore."""
    stem = Path(video_file).stem
    return re.sub(r"[^A-Za-z0-9_-]", "_", stem)


def _format_filename(timestamp_seconds: float, label: str) -> str:
    """t{HHHHH.SS}__{label}.jpg — 5-digit zero-padded seconds, 2-decimal precision."""
    return f"t{int(timestamp_seconds):05d}.{int(round((timestamp_seconds - int(timestamp_seconds)) * 100)):02d}__{label}.jpg"


def build_bundle(
    annotations: list[Annotation],
    frames_root: Path,
    output_root: Path,
    folder_hint: str = "",
) -> Path:
    """
    Compile the bundle:
      1. Per-video folder under output_root.
      2. Each annotation's source frame copied + renamed to the SKILL filename convention.
      3. tacit_annotations_{YYYY-MM-DD}.xlsx with annotations + video_manifest sheets.

    Returns the xlsx path.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. group annotations by video
    by_video: dict[str, list[Annotation]] = {}
    for a in annotations:
        by_video.setdefault(a.video_file, []).append(a)

    # 2. copy + rename frames, populate screenshot_path
    for video_file, anns in by_video.items():
        basename = _basename(video_file)
        video_dir = output_root / basename
        video_dir.mkdir(parents=True, exist_ok=True)
        for ann in anns:
            new_name = _format_filename(ann.timestamp_seconds, ann.your_label)
            src = frames_root / basename / f"t{int(ann.timestamp_seconds):05d}.jpg"
            dst = video_dir / new_name
            if src.exists():
                shutil.copy2(src, dst)
            ann.screenshot_path = f"{basename}/{new_name}"

    # 3. xlsx
    wb = Workbook()
    sheet = wb.active
    sheet.title = "annotations"
    sheet.append(ANNOTATION_COLUMNS)
    for ann in annotations:
        sheet.append(ann.to_row())

    manifest = wb.create_sheet("video_manifest")
    manifest.append(MANIFEST_COLUMNS)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for video_file, anns in by_video.items():
        branch = anns[0].branch
        manifest.append([
            video_file,
            BRANCH_TO_CATEGORY.get(branch, branch),
            branch,
            "done",
            len(anns),
            folder_hint,
            ts,
        ])

    today = _dt.datetime.now().strftime("%Y-%m-%d")
    xlsx_path = output_root / f"tacit_annotations_{today}.xlsx"
    wb.save(xlsx_path)
    return xlsx_path


def annotations_from_json(annotations_json: Path) -> list[Annotation]:
    """Load annotations from the input JSON shape produced by labeling."""
    data = json.loads(annotations_json.read_text())
    branch = data.get("branch", "")
    out = []
    for a in data.get("annotations", []):
        out.append(
            Annotation(
                branch=a.get("branch", branch),
                video_file=a["video_file"],
                timestamp_seconds=float(a.get("timestamp_seconds", a.get("ts", 0))),
                your_label=a.get("your_label", a.get("label", "")),
                confidence=a.get("confidence", "medium"),
                screenshot_path=a.get("screenshot_path", ""),
                substance_tags=a.get("substance_tags", ""),
                action_tags=a.get("action_tags", ""),
                equipment_tags=a.get("equipment_tags", ""),
            )
        )
    return out


def annotations_to_json(annotations: list[Annotation], path: Path, branch: str = "") -> Path:
    """Write annotations to the canonical JSON shape consumed by the bundle/append steps."""
    payload = {
        "branch": branch or (annotations[0].branch if annotations else ""),
        "annotations": [a.to_dict() for a in annotations],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
