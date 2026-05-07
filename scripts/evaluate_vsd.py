"""
evaluate_vsd.py — Run the VSD benchmark (6 confusion pairs).

For each pair of visually similar physical states (defined in
labproc_tacit.data.VSD_PAIRS), train a binary classifier on V-JEPA
features. This isolates fine-grained visual discrimination from text labels.

Usage:
    python scripts/evaluate_vsd.py \\
        --psc-csv /path/to/op_master.csv \\
        --video-root /path/to/videos \\
        --checkpoint /path/to/tacit.pth \\
        --cache /tmp/features_psc_tacit.pt
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
from labproc_tacit.data import load_psc_data
from labproc_tacit.probes import run_vsd


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
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    items = load_psc_data(args.psc_csv, args.video_root)
    feature_cache = extract_or_load_features(
        items, args.checkpoint, args.checkpoint_key, args.cache)

    # Build video -> group ID map for GroupKFold
    video_to_group = {}
    for item in items:
        if item["video_file"] not in video_to_group:
            video_to_group[item["video_file"]] = len(video_to_group)

    results = run_vsd(
        feature_cache, items, video_to_group,
        condition_name=Path(args.checkpoint).stem,
    )

    if results is None:
        sys.exit("ERROR: VSD evaluation produced no usable pair results.")

    print(f"\nVSD Results")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Pairs evaluated: {results['n_pairs']}")
    print(f"  Mean accuracy across pairs: {results['mean_acc']*100:.1f}%")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
