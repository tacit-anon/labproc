"""
evaluate_psc.py — Run the PSC benchmark for one V-JEPA / Tacit checkpoint.

Loads the master annotation CSV, extracts features for every annotated clip
through the given encoder, then runs the PSC GroupKFold probe and prints
overall + per-class accuracy.

Usage:
    python scripts/evaluate_psc.py \\
        --psc-csv /path/to/op_master.csv \\
        --video-root /path/to/videos \\
        --checkpoint /path/to/tacit.pth \\
        --checkpoint-key model_state \\
        --cache /tmp/features_psc_tacit.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from labproc_tacit import (
    DEVICE, build_encoder, encode_video, extract_frames_at_timestamp,
    load_checkpoint,
)
from labproc_tacit.data import LABEL_TO_IDX, load_psc_data
from labproc_tacit.probes import run_psc_groupkfold


def extract_or_load_features(items, checkpoint, checkpoint_key, cache_path):
    """Build a (video_file, timestamp) -> feature cache. Disk-cached."""
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
        print(f"Cached features to {cache_path}", file=sys.stderr)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psc-csv", required=True, type=Path,
                        help="Master OP annotation CSV (op_master.csv)")
    parser.add_argument("--video-root", required=True, type=Path,
                        help="Root directory containing source video files")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to V-JEPA-2.1 base or Tacit checkpoint")
    parser.add_argument("--checkpoint-key", default=None,
                        help="Override checkpoint state-dict key")
    parser.add_argument("--cache", type=Path, default=None,
                        help="Where to cache extracted features (.pt)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write results JSON (default: stdout only)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    items = load_psc_data(args.psc_csv, args.video_root)
    print(f"Loaded {len(items)} PSC items from "
          f"{len(set(it['video_file'] for it in items))} videos",
          file=sys.stderr)

    feature_cache = extract_or_load_features(
        items, args.checkpoint, args.checkpoint_key, args.cache)

    # Stack features and labels in item order
    feats = []
    labels = []
    groups = []
    video_to_group = {}
    for item in items:
        key = (item["video_file"], item["timestamp"])
        if key not in feature_cache:
            continue
        feats.append(feature_cache[key])
        labels.append(LABEL_TO_IDX[item["label"]])
        if item["video_file"] not in video_to_group:
            video_to_group[item["video_file"]] = len(video_to_group)
        groups.append(video_to_group[item["video_file"]])

    features = torch.stack(feats)
    labels_t = torch.tensor(labels, dtype=torch.long)
    groups_a = np.array(groups)

    class_names = sorted(set(item["label"] for item in items),
                          key=lambda x: LABEL_TO_IDX[x])
    # Remap labels to dense indices over the classes actually present
    label_index_remap = {LABEL_TO_IDX[c]: i for i, c in enumerate(class_names)}
    labels_dense = torch.tensor([label_index_remap[int(l)] for l in labels_t],
                                 dtype=torch.long)

    results = run_psc_groupkfold(
        features, labels_dense, groups_a, class_names,
        condition_name=Path(args.checkpoint).stem,
        n_folds=args.n_folds, n_epochs=args.n_epochs, lr=args.lr,
    )

    print(f"\nPSC Results")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Items: {results['n_items']}, Classes: {results['n_classes']}")
    print(f"  Overall accuracy: {results['overall_acc']:.3f}  "
          f"({results['overall_acc']*100:.1f}%)")
    print(f"  Per-fold: {results['fold_accs']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
