"""
LLM-driven labeling backend.

Default backend is the Anthropic API (claude-sonnet-4 or claude-opus-4 family).
The `LabelingClient` interface lets you swap providers (OpenAI Vision, Gemini, local VLM, etc.).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tenacity import retry, stop_after_attempt, wait_exponential

from .extract import parse_timestamp_from_filename
from .prompts import build_triage_user_prompt, build_user_prompt, load_system_prompt
from .schema import Annotation, FrameAuditRecord, TriageDecision


def _encode_image_b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _parse_json_response(text: str) -> dict:
    """Strip common preambles and parse JSON. Tolerates ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        # strip fences
        lines = text.split("\n")
        # drop first fence + optional language tag
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


class LabelingClient(Protocol):
    """Provider-agnostic interface. Implement this to plug a different VLM."""

    def label_frame(self, image_path: Path, system_prompt: str, user_prompt: str) -> dict:
        ...

    def triage_frames(
        self, image_paths: list[Path], system_prompt: str, user_prompt: str
    ) -> dict:
        ...


# ────────────────────────────────────────────────────────────────────────────
# Anthropic backend
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class AnthropicClient:
    """
    Anthropic-API-backed labeling client.

    Reads ANTHROPIC_API_KEY from env if api_key is not passed.
    """

    api_key: str | None = None
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        # Lazy import so non-Anthropic users can still install the package.
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. `pip install anthropic` or use a different backend."
            ) from e
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "no API key. Set ANTHROPIC_API_KEY or pass api_key= explicitly."
            )
        self._client = Anthropic(api_key=key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def label_frame(self, image_path: Path, system_prompt: str, user_prompt: str) -> dict:
        b64 = _encode_image_b64(image_path)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        )
        text = msg.content[0].text  # type: ignore[union-attr]
        return _parse_json_response(text)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def triage_frames(
        self, image_paths: list[Path], system_prompt: str, user_prompt: str
    ) -> dict:
        content: list[dict] = []
        for p in image_paths:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _encode_image_b64(p),
                    },
                }
            )
        content.append({"type": "text", "text": user_prompt})
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        text = msg.content[0].text  # type: ignore[union-attr]
        return _parse_json_response(text)


# ────────────────────────────────────────────────────────────────────────────
# Public functions
# ────────────────────────────────────────────────────────────────────────────


def label_frame(
    image_path: Path,
    branch: str,
    video_file: str,
    timestamp_seconds: int,
    client: LabelingClient,
    repo_root: Path | None = None,
) -> tuple[Annotation | None, FrameAuditRecord]:
    """
    Label a single extracted frame.

    Returns (annotation, audit_record). `annotation` is None if the model chose to skip.
    The audit record is always returned and should be persisted alongside the bundle.
    """
    system_prompt = load_system_prompt(repo_root)
    user_prompt = build_user_prompt(branch, video_file, timestamp_seconds)

    response = client.label_frame(image_path, system_prompt, user_prompt)

    selected = response.get("selected")
    audit = FrameAuditRecord(
        frame=image_path.name,
        timestamp_seconds=float(timestamp_seconds),
        candidates_considered=response.get("candidates_considered", []),
        eliminated_by_apparatus=response.get("eliminated_by_apparatus", ""),
        selected=selected,
        rejected_with_reason=response.get("rejected_with_reason", {}),
        confidence=response.get("confidence", "low"),
    )

    if not selected:
        return None, audit

    annotation = Annotation(
        branch=branch,  # type: ignore[arg-type]
        video_file=video_file,
        timestamp_seconds=float(timestamp_seconds),
        your_label=selected,
        confidence=response.get("confidence", "medium"),  # type: ignore[arg-type]
        screenshot_path="",  # set by build_bundle when frames are renamed
        substance_tags=response.get("substance_tags", ""),
        action_tags=response.get("action_tags", ""),
        equipment_tags=response.get("equipment_tags", ""),
    )
    return annotation, audit


def label_frames_dir(
    frames_dir: Path,
    branch: str,
    video_file: str,
    client: LabelingClient,
    repo_root: Path | None = None,
) -> tuple[list[Annotation], list[FrameAuditRecord]]:
    """Label every `tNNNNN.jpg` in `frames_dir`."""
    annotations: list[Annotation] = []
    audits: list[FrameAuditRecord] = []
    frames = sorted(frames_dir.glob("t*.jpg"))
    for frame in frames:
        ts = parse_timestamp_from_filename(frame)
        ann, aud = label_frame(frame, branch, video_file, ts, client, repo_root)
        audits.append(aud)
        if ann is not None:
            annotations.append(ann)
    return annotations, audits


def triage_video(
    video_file: str,
    branch: str,
    triage_frames: list[Path],
    client: LabelingClient,
    repo_root: Path | None = None,
) -> TriageDecision:
    """Run the dense-triage decision over the extracted triage frames."""
    system_prompt = load_system_prompt(repo_root)
    frames_meta = [(parse_timestamp_from_filename(p), p.name) for p in triage_frames]
    user_prompt = build_triage_user_prompt(branch, video_file, frames_meta)

    response = client.triage_frames(triage_frames, system_prompt, user_prompt)

    return TriageDecision(
        video_file=video_file,
        branch=branch,  # type: ignore[arg-type]
        decision=response.get("decision", "MATCH"),
        reason=response.get("reason", ""),
        label_proposal=response.get("label_proposal", ""),
        frames_inspected=[ts for ts, _ in frames_meta],
    )
