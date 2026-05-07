"""
labproc_tacit — Evaluation utilities for the LabProc benchmark and the Tacit
domain-adapted V-JEPA-2.1 video encoder.

Quick example:

    from labproc_tacit import build_encoder, load_checkpoint, encode_video
    from labproc_tacit.video_io import extract_frames_at_timestamp

    encoder = build_encoder()
    encoder = load_checkpoint(encoder, "/path/to/tacit.pth")
    encoder = encoder.cuda().eval()

    frames = extract_frames_at_timestamp("/path/to/video.mp4", timestamp_sec=86)
    feature = encode_video(encoder, frames)   # 1024-dim tensor

For task-specific evaluation, prefer the scripts in `scripts/`:

    python scripts/evaluate_psc.py --psc-csv ... --video-root ...
"""

from labproc_tacit.encoder import (
    DEVICE,
    build_encoder,
    encode_video,
    load_checkpoint,
)
from labproc_tacit.video_io import (
    WINDOW_SEC,
    extract_frames_at_timestamp,
    find_video_path,
)

__version__ = "1.0.0"

__all__ = [
    "DEVICE",
    "WINDOW_SEC",
    "build_encoder",
    "encode_video",
    "extract_frames_at_timestamp",
    "find_video_path",
    "load_checkpoint",
]
