"""
Data classes representing the pipeline's contracts.

The 10-column annotations schema is the on-disk contract; this module mirrors it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Branch = Literal["op", "pcr", "wb"]
Confidence = Literal["high", "medium", "low"]
Decision = Literal["MATCH", "TAXONOMY_GAP", "NON_CATEGORY_SKIP"]

BRANCH_TO_CATEGORY: dict[str, str] = {
    "op": "organic_purification",
    "pcr": "pcr",
    "wb": "western_blot",
}

ANNOTATION_COLUMNS: list[str] = [
    "branch",
    "video_file",
    "timestamp_seconds",
    "physical_state",
    "confidence",
    "your_label",
    "screenshot_path",
    "substance_tags",
    "action_tags",
    "equipment_tags",
]

MANIFEST_COLUMNS: list[str] = [
    "video_file",
    "category",
    "branch",
    "status",
    "annotation_count",
    "folder",
    "last_modified",
]

AUDIT_COLUMNS: list[str] = [
    "video_file",
    "branch",
    "decision",
    "reason",
    "label_proposal",
    "logged_at",
]


@dataclass(slots=True)
class Annotation:
    """A single labeled frame — one row in the annotations sheet."""

    branch: Branch
    video_file: str
    timestamp_seconds: float
    your_label: str
    confidence: Confidence
    screenshot_path: str
    substance_tags: str = ""
    action_tags: str = ""
    equipment_tags: str = ""

    @property
    def physical_state(self) -> str:
        """Mirrors your_label (legacy schema duplicate kept for tool compatibility)."""
        return self.your_label

    def to_row(self) -> list:
        return [
            self.branch,
            self.video_file,
            float(self.timestamp_seconds),
            self.physical_state,
            self.confidence,
            self.your_label,
            self.screenshot_path,
            self.substance_tags,
            self.action_tags,
            self.equipment_tags,
        ]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["physical_state"] = self.physical_state
        return d

    @property
    def dedup_key(self) -> tuple[str, str, float]:
        return (self.branch, self.video_file, float(self.timestamp_seconds))


@dataclass(slots=True)
class AuditEntry:
    """A skip / taxonomy-gap decision logged at triage time."""

    video_file: str
    branch: Branch
    decision: Decision
    reason: str
    label_proposal: str = ""
    logged_at: str = ""

    @property
    def dedup_key(self) -> tuple[str, str, str, str]:
        return (self.video_file, self.branch, self.decision, self.reason)

    def to_row(self) -> list:
        return [
            self.video_file,
            self.branch,
            self.decision,
            self.reason,
            self.label_proposal,
            self.logged_at,
        ]


@dataclass(slots=True)
class TriageDecision:
    """Per-video triage outcome."""

    video_file: str
    branch: Branch
    decision: Decision
    reason: str = ""
    label_proposal: str = ""
    frames_inspected: list[int] = field(default_factory=list)

    def to_audit_entry(self) -> AuditEntry | None:
        if self.decision == "MATCH":
            return None
        return AuditEntry(
            video_file=self.video_file,
            branch=self.branch,
            decision=self.decision,
            reason=self.reason,
            label_proposal=self.label_proposal,
        )


@dataclass(slots=True)
class FrameAuditRecord:
    """Per-frame anti-anchoring audit JSON, sidecar to the bundle."""

    frame: str
    timestamp_seconds: float
    candidates_considered: list[str]
    eliminated_by_apparatus: str
    selected: str | None
    rejected_with_reason: dict[str, str]
    confidence: str

    def to_dict(self) -> dict:
        return asdict(self)
