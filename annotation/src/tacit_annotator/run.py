"""
End-to-end pipeline orchestration: video -> bundle + master append.

Used by the `tacit-annotate run` CLI; importable for programmatic use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from .append import AppendStats, append_to_master
from .audit import audit_path_for, save_audit_records
from .bundle import _basename, annotations_to_json, build_bundle
from .extract import extract_dense_frames, extract_triage_frames
from .label import LabelingClient, label_frames_dir, triage_video
from .schema import Annotation, FrameAuditRecord, TriageDecision

console = Console()


@dataclass
class PipelineResult:
    video: Path
    triage: TriageDecision
    annotations: list[Annotation]
    audits: list[FrameAuditRecord]
    bundle_xlsx: Path | None
    audit_json: Path | None
    append_stats: AppendStats | None


def run_pipeline(
    video: Path,
    branch: str,
    output_dir: Path,
    client: LabelingClient,
    interval: int = 30,
    triage_n: int = 10,
    master_xlsx: Path | None = None,
    folder_hint: str = "",
    skip_triage: bool = False,
    repo_root: Path | None = None,
) -> PipelineResult:
    """
    1. Extract triage frames -> triage decision (unless skip_triage).
    2. If MATCH, extract dense frames at `interval` and label them.
    3. Save audit JSON.
    4. Build bundle (xlsx + renamed frames).
    5. Optionally append to master corpus.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = _basename(video.name)

    triage_dir = output_dir / "triage_frames" / basename
    dense_dir = output_dir / "dense_frames" / basename
    bundle_root = output_dir / "bundle"

    # 1. Triage
    if skip_triage:
        triage = TriageDecision(video_file=video.name, branch=branch, decision="MATCH")  # type: ignore[arg-type]
    else:
        console.log(f"[cyan]triage[/cyan] {video.name}")
        triage_frames = extract_triage_frames(video, triage_dir, n=triage_n)
        triage = triage_video(video.name, branch, triage_frames, client, repo_root)
        console.log(f"  decision={triage.decision} reason={triage.reason!r}")

    if triage.decision != "MATCH":
        # Persist a minimal audit; no labeling.
        audit_entry = triage.to_audit_entry()
        result = PipelineResult(
            video=video,
            triage=triage,
            annotations=[],
            audits=[],
            bundle_xlsx=None,
            audit_json=None,
            append_stats=None,
        )
        if master_xlsx is not None and audit_entry is not None:
            ann_json = output_dir / f"triage_{basename}.json"
            ann_json.write_text(
                '{"branch":"' + branch + '","annotations":[],"audit":[' +
                str(audit_entry.__dict__).replace("'", '"') + "]}"
            )
            stats = append_to_master(ann_json, master_xlsx, folder_hint=folder_hint)
            result.append_stats = stats
        return result

    # 2. Dense extraction + labeling
    console.log(f"[cyan]extract[/cyan] dense frames at {interval}s intervals")
    extract_dense_frames(video, dense_dir, interval=interval)

    console.log("[cyan]label[/cyan] frames")
    annotations: list[Annotation] = []
    audits: list[FrameAuditRecord] = []
    frames = sorted(dense_dir.glob("t*.jpg"))

    with Progress() as progress:
        task = progress.add_task("labeling...", total=len(frames))
        anns, auds = label_frames_dir(dense_dir, branch, video.name, client, repo_root)
        annotations.extend(anns)
        audits.extend(auds)
        progress.update(task, advance=len(frames))

    yield_pct = (len(annotations) / max(len(frames), 1)) * 100
    console.log(
        f"  {len(annotations)}/{len(frames)} frames labeled "
        f"({yield_pct:.0f}% yield, {len(frames) - len(annotations)} skipped)"
    )

    # 3. Audit JSON
    audit_json = save_audit_records(audits, audit_path_for(basename, output_dir))
    console.log(f"  audit -> {audit_json}")

    # 4. Bundle
    annotations_json = output_dir / f"annotations_{basename}.json"
    annotations_to_json(annotations, annotations_json, branch=branch)

    # frames input root for build_bundle is the parent of dense_dir
    bundle_xlsx = build_bundle(
        annotations,
        frames_root=output_dir / "dense_frames",
        output_root=bundle_root,
        folder_hint=folder_hint,
    )
    console.log(f"[green]bundle[/green] -> {bundle_xlsx}")

    # 5. Master append
    append_stats = None
    if master_xlsx is not None:
        append_stats = append_to_master(annotations_json, master_xlsx, folder_hint=folder_hint)
        console.log(
            f"[green]master[/green] +{append_stats.appended} rows "
            f"({append_stats.skipped_duplicates} dup), "
            f"CSV: {append_stats.csv_counts}"
        )

    return PipelineResult(
        video=video,
        triage=triage,
        annotations=annotations,
        audits=audits,
        bundle_xlsx=bundle_xlsx,
        audit_json=audit_json,
        append_stats=append_stats,
    )
