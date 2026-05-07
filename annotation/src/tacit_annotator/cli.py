"""
Tacit Video Annotator — command-line interface.

Subcommands:
    triage    Run dense-sampling triage on a video (10 frames + decision).
    extract   Extract dense frames at fixed interval.
    label     Label frames in a directory using the LLM backend.
    build     Compile a bundle from annotations.json + frames.
    append    Append a bundle/annotations.json to a master corpus xlsx.
    run       Full end-to-end pipeline for a single video.
    batch     Full end-to-end pipeline for every video in a folder.
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .__version__ import __version__
from .append import append_to_master
from .audit import audit_path_for, save_audit_records
from .bundle import _basename, annotations_from_json, annotations_to_json, build_bundle
from .extract import extract_dense_frames, extract_triage_frames
from .label import AnthropicClient, label_frames_dir, triage_video
from .run import run_pipeline

console = Console()


def _make_client(model: str) -> AnthropicClient:
    return AnthropicClient(model=model)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="tacit-annotate")
def main() -> None:
    """Tacit Video Annotator — hierarchical labeling for laboratory procedure videos."""


# ────────────────────────────────────────────────────────────────────────────
# triage
# ────────────────────────────────────────────────────────────────────────────


@main.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--branch", required=True, type=click.Choice(["op", "pcr", "wb"]))
@click.option("--output-dir", default="./out", type=click.Path(path_type=Path), help="Where to write triage frames + decision.")
@click.option("--n", default=10, show_default=True, help="Number of triage frames to extract.")
@click.option("--model", default="claude-sonnet-4-6", show_default=True)
def triage(video: Path, branch: str, output_dir: Path, n: int, model: str) -> None:
    """Run dense-sampling triage on VIDEO. Outputs frames + a triage decision."""
    output_dir.mkdir(parents=True, exist_ok=True)
    triage_dir = output_dir / "triage_frames" / _basename(video.name)
    console.log(f"extracting {n} triage frames -> {triage_dir}")
    frames = extract_triage_frames(video, triage_dir, n=n)
    client = _make_client(model)
    decision = triage_video(video.name, branch, frames, client)
    console.print(decision)


# ────────────────────────────────────────────────────────────────────────────
# extract
# ────────────────────────────────────────────────────────────────────────────


@main.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--interval", default=30, show_default=True, help="Seconds between frames.")
def extract(video: Path, output_dir: Path, interval: int) -> None:
    """Extract dense frames from VIDEO into OUTPUT_DIR at fixed interval."""
    paths = extract_dense_frames(video, output_dir, interval=interval)
    console.log(f"extracted {len(paths)} frames -> {output_dir}")


# ────────────────────────────────────────────────────────────────────────────
# label
# ────────────────────────────────────────────────────────────────────────────


@main.command(name="label")
@click.argument("frames_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--branch", required=True, type=click.Choice(["op", "pcr", "wb"]))
@click.option("--video-file", required=True, help="Original video filename (for the annotations row).")
@click.option("--output", default="./annotations.json", type=click.Path(path_type=Path))
@click.option("--audit-output", default=None, type=click.Path(path_type=Path), help="Audit JSON path. Defaults to audit_{basename}.json next to --output.")
@click.option("--model", default="claude-sonnet-4-6", show_default=True)
def label_cmd(frames_dir: Path, branch: str, video_file: str, output: Path, audit_output: Path | None, model: str) -> None:
    """Label every frame in FRAMES_DIR. Writes annotations + audit JSONs."""
    client = _make_client(model)
    annotations, audits = label_frames_dir(frames_dir, branch, video_file, client)
    annotations_to_json(annotations, output, branch=branch)
    audit_target = audit_output or output.parent / f"audit_{_basename(video_file)}.json"
    save_audit_records(audits, audit_target)

    yield_pct = (len(annotations) / max(len(audits), 1)) * 100
    table = Table(show_header=True)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("frames", str(len(audits)))
    table.add_row("labeled", str(len(annotations)))
    table.add_row("skipped", str(len(audits) - len(annotations)))
    table.add_row("yield", f"{yield_pct:.0f}%")
    console.print(table)


# ────────────────────────────────────────────────────────────────────────────
# build
# ────────────────────────────────────────────────────────────────────────────


@main.command()
@click.argument("annotations_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_root", type=click.Path(path_type=Path))
@click.option("--frames-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Directory containing per-video frame subfolders.")
@click.option("--folder-hint", default="", help="Source folder for the manifest sheet.")
def build(annotations_json: Path, output_root: Path, frames_root: Path, folder_hint: str) -> None:
    """Compile a Tacit-import-ready bundle from ANNOTATIONS_JSON."""
    annotations = annotations_from_json(annotations_json)
    xlsx = build_bundle(annotations, frames_root=frames_root, output_root=output_root, folder_hint=folder_hint)
    console.log(f"bundle -> {xlsx}")


# ────────────────────────────────────────────────────────────────────────────
# append
# ────────────────────────────────────────────────────────────────────────────


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("master_xlsx", type=click.Path(path_type=Path))
@click.option("--folder", default="", help="Source folder hint for the manifest.")
def append(input_path: Path, master_xlsx: Path, folder: str) -> None:
    """Idempotently append INPUT_PATH (json or xlsx) into MASTER_XLSX."""
    stats = append_to_master(input_path, master_xlsx, folder_hint=folder)
    table = Table(show_header=True)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("appended", str(stats.appended))
    table.add_row("dup-skipped", str(stats.skipped_duplicates))
    table.add_row("audit-added", str(stats.audit_added))
    table.add_row("audit-dup-skipped", str(stats.audit_skipped_duplicates))
    table.add_row("CSV exports", ", ".join(f"{k}={v}" for k, v in stats.csv_counts.items()))
    console.print(table)


# ────────────────────────────────────────────────────────────────────────────
# run (single video, end-to-end)
# ────────────────────────────────────────────────────────────────────────────


@main.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--branch", required=True, type=click.Choice(["op", "pcr", "wb"]))
@click.option("--output-dir", default="./out", type=click.Path(path_type=Path))
@click.option("--master-xlsx", default=None, type=click.Path(path_type=Path), help="Append to this master corpus.")
@click.option("--interval", default=30, show_default=True)
@click.option("--triage-n", default=10, show_default=True)
@click.option("--skip-triage", is_flag=True, help="Skip the validity-filter triage step.")
@click.option("--folder-hint", default="")
@click.option("--model", default="claude-sonnet-4-6", show_default=True)
def run(
    video: Path,
    branch: str,
    output_dir: Path,
    master_xlsx: Path | None,
    interval: int,
    triage_n: int,
    skip_triage: bool,
    folder_hint: str,
    model: str,
) -> None:
    """Full end-to-end pipeline on VIDEO: triage -> extract -> label -> bundle -> append."""
    client = _make_client(model)
    result = run_pipeline(
        video=video,
        branch=branch,
        output_dir=output_dir,
        client=client,
        interval=interval,
        triage_n=triage_n,
        master_xlsx=master_xlsx,
        folder_hint=folder_hint,
        skip_triage=skip_triage,
    )

    table = Table(show_header=True)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("triage decision", result.triage.decision)
    table.add_row("annotations", str(len(result.annotations)))
    table.add_row("frames inspected", str(len(result.audits)))
    if result.bundle_xlsx:
        table.add_row("bundle", str(result.bundle_xlsx))
    if result.audit_json:
        table.add_row("audit", str(result.audit_json))
    console.print(table)


# ────────────────────────────────────────────────────────────────────────────
# batch (folder of videos, end-to-end)
# ────────────────────────────────────────────────────────────────────────────


@main.command()
@click.argument("video_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--branch", required=True, type=click.Choice(["op", "pcr", "wb"]))
@click.option("--output-dir", default="./out", type=click.Path(path_type=Path))
@click.option("--master-xlsx", default=None, type=click.Path(path_type=Path))
@click.option("--interval", default=30, show_default=True)
@click.option("--triage-n", default=10, show_default=True)
@click.option("--folder-hint", default="")
@click.option("--quality-tier", default="good_yt", show_default=True, help="Filename prefix to filter on.")
@click.option("--model", default="claude-sonnet-4-6", show_default=True)
def batch(
    video_dir: Path,
    branch: str,
    output_dir: Path,
    master_xlsx: Path | None,
    interval: int,
    triage_n: int,
    folder_hint: str,
    quality_tier: str,
    model: str,
) -> None:
    """Run the pipeline for every `{quality_tier}_*.mp4` in VIDEO_DIR."""
    client = _make_client(model)
    videos = sorted(video_dir.glob(f"{quality_tier}_*.mp4"))
    console.log(f"batch: {len(videos)} videos in {video_dir} (tier={quality_tier})")

    summary: list[tuple[str, str, int]] = []
    for video in videos:
        try:
            result = run_pipeline(
                video=video,
                branch=branch,
                output_dir=output_dir,
                client=client,
                interval=interval,
                triage_n=triage_n,
                master_xlsx=master_xlsx,
                folder_hint=folder_hint or str(video_dir),
            )
            summary.append((video.name, result.triage.decision, len(result.annotations)))
        except Exception as e:  # noqa: BLE001
            console.log(f"[red]error[/red] on {video.name}: {e}")
            summary.append((video.name, "ERROR", 0))

    table = Table(title="Batch summary")
    table.add_column("video")
    table.add_column("decision")
    table.add_column("rows", justify="right")
    for name, decision, count in summary:
        table.add_row(name, decision, str(count))
    console.print(table)


if __name__ == "__main__":
    main()
