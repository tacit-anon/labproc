"""
evaluate_ccr.py — Run the standard CCR benchmark (288 pairs / 20 groups).

Constructs CCR groups from the master annotation CSV (one group per video
with >=4 distinct-state clips), then runs leave-one-video-out pairwise
ordering evaluation.

Usage:
    python scripts/evaluate_ccr.py \\
        --psc-csv /path/to/op_master.csv \\
        --video-root /path/to/videos \\
        --checkpoint /path/to/tacit.pth \\
        --checkpoint-key model_state \\
        --cache /tmp/features_psc_tacit.pt   # can reuse PSC cache
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
from labproc_tacit.data import construct_ccr_groups, load_psc_data
from labproc_tacit.probes import run_ccr_leave_one_out


def extract_or_load_features(items, checkpoint, checkpoint_key, cache_path):
    if cache_path and cache_path.exists():
        print(f"Loading cached features from {cache_path}", file=sys.stderr)
        return torch.load(cache_path, map_location="cpu")

    print(f"Loading encoder from {checkpoint}", file=sys.stderr)
    encoder = build_encoder()
    encoder = load_checkpoint(encoder, checkpoint, key=checkpoint_key)
    encoder = encoder.to(DEVICE).eval()

    cache = {}
    t0 = time.time()
    for i, item in enumerate(items):
        key = (item["video_file"], item["timestamp"])
        if key in cache:
            continue
        frames = extract_frames_at_timestamp(item["video_path"], item["timestamp"])
        cache[key] = encode_video(encoder, frames)
        if (i + 1) % 25 == 0:
            print(f"  encoded {i+1}/{len(items)} ({time.time()-t0:.0f}s)",
                  file=sys.stderr)

    del encoder
    torch.cuda.empty_cache()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psc-csv", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--checkpoint-key", default=None)
    parser.add_argument("--cache", type=Path, default=None,
                        help="Feature cache (compatible with the PSC script)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    items = load_psc_data(args.psc_csv, args.video_root)
    feature_cache = extract_or_load_features(
        items, args.checkpoint, args.checkpoint_key, args.cache)

    groups = construct_ccr_groups(items)
    print(f"Constructed {len(groups)} CCR groups", file=sys.stderr)

    # Build per-group feature lists, dropping clips whose features are missing
    group_features = []
    filtered_groups = []
    for group in groups:
        clip_feats = []
        for item in group["items"]:
            key = (item["video_file"], item["timestamp"])
            if key in feature_cache:
                clip_feats.append(feature_cache[key])
        if len(clip_feats) >= 4:
            group_features.append(clip_feats)
            filtered_groups.append(group)

    if len(filtered_groups) < 3:
        sys.exit(f"ERROR: only {len(filtered_groups)} groups have features, "
                 f"need >=3 for leave-one-out CV.")

    results = run_ccr_leave_one_out(
        group_features, filtered_groups,
        condition_name=Path(args.checkpoint).stem,
    )

    print(f"\nCCR Results")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Groups: {results['n_groups']}, "
          f"Total pairs: {results['n_total_pairs']}")
    print(f"  Mean Kendall's tau: {results['mean_tau']:+.3f} "
          f"± {results['std_tau']:.3f}")
    print(f"  Mean pairwise accuracy: "
          f"{results['mean_pairwise_acc']:.3f}  "
          f"({results['mean_pairwise_acc']*100:.1f}%)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
