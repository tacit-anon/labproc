"""
Audit-trail persistence — the anti-anchoring sidecar JSONs.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import FrameAuditRecord


def save_audit_records(records: list[FrameAuditRecord], out_path: Path) -> Path:
    """Write a list of FrameAuditRecord to a single JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "frame_count": len(records),
        "skipped_count": sum(1 for r in records if r.selected is None),
        "labeled_count": sum(1 for r in records if r.selected is not None),
        "records": [r.to_dict() for r in records],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def audit_path_for(video_basename: str, output_root: Path) -> Path:
    return output_root / f"audit_{video_basename}.json"
