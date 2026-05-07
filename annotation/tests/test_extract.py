"""Frame extraction unit tests. Skipped in environments without ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tacit_annotator.extract import (
    extract_dense_frames,
    extract_triage_frames,
    parse_timestamp_from_filename,
)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    """Generate a 60-second test video using ffmpeg's testsrc."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")
    out = tmp_path / "dummy.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=60:size=320x240:rate=10",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            str(out),
        ],
        check=True,
    )
    return out


def test_dense_extraction(dummy_video, tmp_path):
    out_dir = tmp_path / "frames"
    paths = extract_dense_frames(dummy_video, out_dir, interval=10)
    # 60s video at 10s intervals → t=0, 10, 20, 30, 40, 50, 60 → 7 frames
    assert len(paths) == 7
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert paths[0].name == "t00000.jpg"


def test_triage_extraction(dummy_video, tmp_path):
    out_dir = tmp_path / "triage"
    paths = extract_triage_frames(dummy_video, out_dir, n=10)
    assert len(paths) == 10
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_parse_timestamp_from_filename():
    assert parse_timestamp_from_filename(Path("t00150.jpg")) == 150
    assert parse_timestamp_from_filename(Path("/abs/t00000.jpg")) == 0
