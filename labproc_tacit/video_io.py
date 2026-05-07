"""
labproc_tacit/video_io.py — Video decoding and path resolution.

Pulled out of the per-task scripts so all evaluation paths use the same
clip-window and normalization conventions. If you change WINDOW_SEC or the
ImageNet normalization stats here, every downstream evaluation will pick
up the change consistently.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

# 4 seconds was selected during development as the best window from a
# directional sensitivity test (vs 2s and 8s alternatives).
WINDOW_SEC = 4.0

# ImageNet normalization (matches V-JEPA-2.1 pretraining)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def extract_frames_at_timestamp(
    video_path: str | Path,
    timestamp_sec: float,
    *,
    n_frames: int = 16,
    frame_size: int = 384,
    window_sec: float = WINDOW_SEC,
) -> torch.Tensor:
    """
    Decode `n_frames` frames evenly spaced across a `window_sec` window
    centered at `timestamp_sec`. Returns a (3, n_frames, frame_size,
    frame_size) float tensor with ImageNet normalization applied.

    Notes:
      * V-JEPA-2.1 was trained with num_frames=64 but supports variable
        frame counts at inference via interpolated RoPE positions. We use
        n_frames=16 by default at inference time, which matches the train
        scripts in the original codebase.
      * Frames that fail to decode are replaced with zeros (a defensive
        fallback; in practice this rarely happens with well-encoded MP4s).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return torch.zeros(3, n_frames, frame_size, frame_size)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps

    start_sec = max(0.0, timestamp_sec - window_sec / 2.0)
    end_sec = min(total_sec, timestamp_sec + window_sec / 2.0)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    if end_frame <= start_frame:
        end_frame = min(start_frame + n_frames, total_frames)

    indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size, frame_size))
        frames.append(frame)
    cap.release()

    arr = np.stack(frames).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return torch.from_numpy(arr).float()


def find_video_path(video_file: str, video_root: str | Path) -> str | None:
    """
    Locate `video_file` somewhere under `video_root`. Returns the absolute
    path as a string, or None if not found.

    Searches the root directory first, then any immediate subdirectory.
    Useful when source videos are organized into per-source subfolders
    (e.g., `youtube/`, `pmc-jove/`).
    """
    root = Path(video_root)

    direct = root / video_file
    if direct.exists():
        return str(direct)

    for sub in root.iterdir():
        if sub.is_dir():
            cand = sub / video_file
            if cand.exists():
                return str(cand)

    return None
