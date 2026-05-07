"""
evaluate_ted_visual.py — Run the TED-Visual triplet benchmark.

Reports overall, per-difficulty, Hard, and Strict-Hard accuracy. Also reports
parameter-free baselines (raw cosine, projected cosine, anchor ablation) for
sanity-checking against the trained siamese probe.

Usage:
    python scripts/evaluate_ted_visual.py \\
        --triplets-csv /path/to/ted_visual.csv \\
        --video-root /path/to/videos \\
        --checkpoint /path/to/tacit.pth \\
        --cache /tmp/features_ted_visual_tacit.pt
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
from labproc_tacit.data import load_ted_visual_data
from labproc_tacit.probes.ted_visual import (
    summarize_triplet_results, train_eval_siamese_groupkfold,
)


def extract_features_for_triplets(triplets, checkpoint, checkpoint_key,
                                    cache_path):
    """Each triplet has 3 clips: anchor, correct, distractor. Cache by
    (video_file, timestamp) so identical clips referenced by multiple
    triplets are encoded once."""
    if cache_path and cache_path.exists():
        print(f"Loading cached features from {cache_path}", file=sys.stderr)
        return torch.load(cache_path, map_location="cpu")

    print(f"Loading encoder from {checkpoint}", file=sys.stderr)
    encoder = build_encoder()
    encoder = load_checkpoint(encoder, checkpoint, key=checkpoint_key)
    encoder = encoder.to(DEVICE).eval()

    cache = {}
    t0 = time.time()
    n_clips_total = len(triplets) * 3
    n_done = 0
    for t in triplets:
        for slot in ("anchor", "correct", "distractor"):
            path_key = f"{slot}_path" if slot == "anchor" else f"{slot}_path"
            ts_key = f"{slot}_ts"
            video_path = t[path_key]
            ts = t[ts_key]
            key = (video_path, ts)
            if key not in cache:
                frames = extract_frames_at_timestamp(video_path, ts)
                cache[key] = encode_video(encoder, frames)
            n_done += 1
            if n_done % 30 == 0:
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
    parser.add_argument("--triplets-csv", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--checkpoint-key", default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--n-epochs", type=int, default=200)
    args = parser.parse_args()

    triplets = load_ted_visual_data(args.triplets_csv, args.video_root)
    print(f"Loaded {len(triplets)} triplets", file=sys.stderr)

    feature_cache = extract_features_for_triplets(
        triplets, args.checkpoint, args.checkpoint_key, args.cache)

    # Attach features to each triplet
    triplets_data = []
    for t in triplets:
        triplets_data.append({
            "anchor_video": t["anchor_video"],
            "difficulty": t["difficulty"],
            "strict_hard": t["strict_hard"],
            "anchor_feat": feature_cache[(t["anchor_path"], t["anchor_ts"])],
            "correct_feat": feature_cache[(t["correct_path"], t["correct_ts"])],
            "distractor_feat": feature_cache[
                (t["distractor_path"], t["distractor_ts"])],
        })

    arrays = train_eval_siamese_groupkfold(
        triplets_data, n_epochs=args.n_epochs, margin=args.margin,
        condition_name=Path(args.checkpoint).stem,
    )

    if arrays is None:
        sys.exit("ERROR: TED-Visual evaluation could not run "
                 "(too few groups for cross-validation).")

    results = {
        "siamese": summarize_triplet_results(arrays["siamese"], triplets_data),
        "cos_raw": summarize_triplet_results(arrays["cos_raw"], triplets_data),
        "anchor_ablation": summarize_triplet_results(
            arrays["anchor_ablation"], triplets_data),
    }

    print(f"\nTED-Visual Results")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Triplets: {len(triplets)}")
    for k, v in results.items():
        line = f"  {k:>16s}: overall={v['overall']*100:5.1f}%"
        if "hard" in v:
            line += f"  hard={v['hard']['accuracy']*100:5.1f}% (n={v['hard']['n']})"
        if "hard_strict" in v:
            line += f"  strict_hard={v['hard_strict']['accuracy']*100:5.1f}% " \
                    f"(n={v['hard_strict']['n']})"
        print(line)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
