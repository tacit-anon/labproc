"""
labproc_tacit/encoder.py — V-JEPA-2.1 ViT-Large encoder utilities.

This module is the single source of truth for encoder construction, checkpoint
loading, and feature extraction. It replaces the 5+ duplicate copies of these
functions that were previously scattered across train_*.py files.

The V-JEPA-2.1 architecture is loaded from the official Meta repository,
which must be cloned to a location on the Python path. See the top-level
README for installation instructions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Where the V-JEPA-2 repo's `app/` module lives. Override via the
# VJEPA2_REPO_ROOT environment variable if you cloned it somewhere else.
DEFAULT_VJEPA2_REPO_ROOT = os.environ.get(
    "VJEPA2_REPO_ROOT",
    str(Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_vjepa2_main"),
)


def _ensure_vjepa2_on_path() -> None:
    """Insert the V-JEPA-2 repo root into sys.path if not already present."""
    if DEFAULT_VJEPA2_REPO_ROOT not in sys.path:
        if not Path(DEFAULT_VJEPA2_REPO_ROOT).exists():
            raise RuntimeError(
                f"V-JEPA-2 repo not found at {DEFAULT_VJEPA2_REPO_ROOT}.\n"
                f"Clone it with:\n"
                f"  git clone https://github.com/facebookresearch/jepa "
                f"{DEFAULT_VJEPA2_REPO_ROOT}\n"
                f"Or set the VJEPA2_REPO_ROOT environment variable to its location."
            )
        sys.path.insert(0, DEFAULT_VJEPA2_REPO_ROOT)


def build_encoder() -> torch.nn.Module:
    """
    Build the V-JEPA-2.1 ViT-Large encoder with the same configuration used
    in the paper. The model is returned uninitialized; call `load_checkpoint`
    next to load weights.
    """
    _ensure_vjepa2_on_path()
    from app.vjepa_2_1.models import vision_transformer as vit_encoder

    return vit_encoder.__dict__["vit_large"](
        patch_size=16,
        img_size=(384, 384),
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )


def load_checkpoint(
    encoder: torch.nn.Module,
    checkpoint_path: str | Path,
    key: str | None = None,
) -> torch.nn.Module:
    """
    Load weights from a checkpoint file into the given encoder.

    Handles three checkpoint formats automatically:
      * Tacit-style adapted checkpoints stored under `model_state`
      * Base V-JEPA-2.1 checkpoints stored under `ema_encoder` (with
        `module.` and `backbone.` prefixes that need stripping)
      * Bare state-dict files

    Pass `key` to override the auto-detected dict key.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    if key and isinstance(ckpt, dict) and key in ckpt:
        state = ckpt[key]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "ema_encoder" in ckpt:
        state = ckpt["ema_encoder"]
        state = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state.items()
        }
    else:
        state = ckpt

    encoder.load_state_dict(state, strict=False)
    return encoder


def encode_video(
    encoder: torch.nn.Module,
    frames_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Run a single clip through the encoder and return its mean-pooled
    feature vector.

    `frames_tensor` is expected to have shape (3, T, H, W) with normalized
    pixel values; see `video_io.extract_frames_at_timestamp` for the canonical
    way to produce it. Returns a (1024,) CPU float tensor.
    """
    x = frames_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        out = encoder(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        feat = out.mean(dim=1).squeeze(0).cpu().float()
    return feat
