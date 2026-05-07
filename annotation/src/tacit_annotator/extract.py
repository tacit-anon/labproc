"""
Frame extraction. Always uses ffmpeg -ss for exact-timestamp seek.
NEVER uses the fps= filter (drifts and silently corrupts dataset alignment).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegError("ffmpeg not found in PATH. Install with `brew install ffmpeg` or `apt install ffmpeg`.")
    return path


def _require_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise FFmpegError("ffprobe not found in PATH (ships with ffmpeg).")
    return path


def get_duration_seconds(video: Path) -> float:
    """Return the duration of `video` in seconds, via ffprobe."""
    _require_ffprobe()
    if not video.exists():
        raise FileNotFoundError(video)
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout.strip()
    if not out:
        raise FFmpegError(f"could not determine duration for {video}")
    return float(out)


def extract_frame_at(video: Path, timestamp_seconds: float, out_path: Path) -> Path:
    """
    Extract a single frame at exact timestamp. Uses -ss BEFORE -i for fast seek.
    Output is JPEG at native resolution, quality 2 (high).
    """
    _require_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss", f"{timestamp_seconds:.3f}",
            "-i", str(video),
            "-frames:v", "1",
            "-q:v", "2",
            "-loglevel", "error",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def extract_dense_frames(
    video: Path,
    out_dir: Path,
    interval: int = 30,
) -> list[Path]:
    """
    Extract frames at fixed intervals (default 30s) from t=0 to t=duration.
    Output filenames: t{seconds_padded5}.jpg

    Returns list of output paths, sorted by timestamp.
    """
    duration = int(get_duration_seconds(video))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    ts = 0
    while ts <= duration:
        out = out_dir / f"t{ts:05d}.jpg"
        extract_frame_at(video, ts, out)
        paths.append(out)
        ts += interval
    return paths


def extract_triage_frames(
    video: Path,
    out_dir: Path,
    n: int = 10,
) -> list[Path]:
    """
    Extract `n` evenly-spaced frames at timestamps `i * duration / (n+1)` for i in 1..n.
    Covers approximately 9–91% of the timeline (for n=10).

    Output filenames: t{seconds_padded5}.jpg
    """
    duration = get_duration_seconds(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(1, n + 1):
        ts = i * duration / (n + 1)
        ts_int = int(round(ts))
        out = out_dir / f"t{ts_int:05d}.jpg"
        extract_frame_at(video, ts, out)
        paths.append(out)
    return paths


def parse_timestamp_from_filename(path: Path) -> int:
    """Parse `t00150.jpg` -> 150."""
    stem = path.stem
    if not stem.startswith("t"):
        raise ValueError(f"unexpected frame filename: {path.name}")
    return int(stem[1:].split(".")[0])
