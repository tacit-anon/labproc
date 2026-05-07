"""
evaluate_ccr_same_state.py — Same-State CCR pilot evaluation.

NOTE: This is NOT part of the v1 LabProc benchmark. The released Tacit
checkpoint is not the appropriate evaluation target for Same-State CCR;
the adaptation pipeline attenuates within-state temporal coherence (see
companion paper Section 6). This script is provided for users investigating
alternative adaptation strategies.

Usage:
    python scripts/evaluate_ccr_same_state.py \\
        --groups-csv /path/to/ccr_same_state.csv \\
        --video-root /path/to/videos \\
        --checkpoint /path/to/some_other_adapted_v-jepa.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from labproc_tacit import (
    DEVICE, build_encoder, encode_video, extract_frames_at_timestamp,
    load_checkpoint,
)
from labproc_tacit.data import load_same_state_ccr_groups
from labproc_tacit.probes.ccr_same_state import evaluate_leave_one_out


def extract_features_for_groups(groups, checkpoint, checkpoint_key, cache_path):
    if cache_path and cache_path.exists():
        print(f"Loading cached features from {cache_path}", file=sys.stderr)
        return torch.load(cache_path, map_location="cpu")

    print(f"Loading encoder from {checkpoint}", file=sys.stderr)
    encoder = build_encoder()
    encoder = load_checkpoint(encoder, checkpoint, key=checkpoint_key)
    encoder = encoder.to(DEVICE).eval()

    cache = {}
    t0 = time.time()
    n_clips_total = sum(len(g["timestamps"]) for g in groups)
    n_done = 0
    for group in groups:
        for ts in group["timestamps"]:
            key = (group["video_path"], ts)
            if key not in cache:
                frames = extract_frames_at_timestamp(group["video_path"], ts)
                cache[key] = encode_video(encoder, frames)
            n_done += 1
            if n_done % 25 == 0:
                print(f"  encoded {n_done}/{n_clips_total} clips "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr)

    del encoder
    torch.cuda.empty_cache()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups-csv", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--checkpoint-key", default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-epochs", type=int, default=100)
    args = parser.parse_args()

    groups = load_same_state_ccr_groups(args.groups_csv, args.video_root)
    print(f"Loaded {len(groups)} same-state groups", file=sys.stderr)

    if len(groups) < 3:
        sys.exit(f"ERROR: only {len(groups)} groups; need >=3 for leave-one-out.")

    feature_cache = extract_features_for_groups(
        groups, args.checkpoint, args.checkpoint_key, args.cache)

    group_features = []
    group_states = []
    for group in groups:
        feats = [feature_cache[(group["video_path"], ts)]
                 for ts in group["timestamps"]]
        group_features.append(feats)
        group_states.append(group["state"])

    results = evaluate_leave_one_out(
        group_features, group_states,
        condition_name=Path(args.checkpoint).stem,
        n_epochs=args.n_epochs,
    )

    print(f"\nSame-State CCR Results")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Groups: {len(groups)}")
    print(f"  Mean Kendall's tau: {results['mean_tau']:+.3f} "
          f"± {results['std_tau']:.3f}")
    print(f"  Mean pairwise accuracy: {results['mean_pair_acc']*100:.1f}% "
          f"± {results['std_pair_acc']*100:.1f}%")
    print(f"  (Random baselines: tau=0.000, pair_acc=50.0%)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
