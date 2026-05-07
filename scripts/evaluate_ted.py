"""
evaluate_ted.py — Run the TED benchmark (4-way MCQ).

Two probe variants:
  --variant visual_only:  PSC classifier scored on TED option labels
  --variant visual_text:  V-JEPA visual + CLIP text fused via cross-attended scorer
  --variant both:         run both, print both numbers

The visual_text variant requires the CLIP package: `pip install clip-by-openai`
or `pip install git+https://github.com/openai/CLIP.git`.

Usage:
    python scripts/evaluate_ted.py \\
        --psc-csv /path/to/op_master.csv \\
        --ted-csv /path/to/ted_op.csv \\
        --video-root /path/to/videos \\
        --checkpoint /path/to/tacit.pth \\
        --variant both
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
from labproc_tacit.data import LABEL_TO_IDX, VALID_OP_LABELS, load_psc_data, load_ted_data
from labproc_tacit.probes.ted import (
    build_text_encoder, encode_text,
    evaluate_visual_only_ted, train_visual_only_psc,
    train_and_eval_visual_text_groupkfold,
)


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


def run_visual_only(psc_features, psc_labels, ted_features, ted_items):
    n_classes = len(VALID_OP_LABELS)
    probe = train_visual_only_psc(psc_features, psc_labels, n_classes)

    ted_options_indices = torch.tensor(
        [t["option_indices"] for t in ted_items], dtype=torch.long)
    ted_correct = torch.tensor(
        [t["correct_idx"] for t in ted_items], dtype=torch.long)

    acc = evaluate_visual_only_ted(probe, ted_features, ted_options_indices,
                                    ted_correct)
    return {"variant": "visual_only", "accuracy": round(acc, 3),
            "n_items": len(ted_items)}


def run_visual_text(ted_features, ted_items):
    print("Building CLIP text encoder...", file=sys.stderr)
    clip_model = build_text_encoder()
    questions = [t["question"] for t in ted_items]
    options_flat = []
    for t in ted_items:
        options_flat.extend(t["options"])

    print("Encoding TED text features...", file=sys.stderr)
    q_feats = encode_text(clip_model, questions)        # (N, 768)
    o_feats = encode_text(clip_model, options_flat).reshape(
        len(ted_items), 4, -1)                          # (N, 4, 768)
    correct_idx = torch.tensor([t["correct_idx"] for t in ted_items],
                                dtype=torch.long)

    # Group by source video
    video_to_group = {}
    groups = []
    for t in ted_items:
        if t["video_file"] not in video_to_group:
            video_to_group[t["video_file"]] = len(video_to_group)
        groups.append(video_to_group[t["video_file"]])
    groups = np.array(groups)

    return train_and_eval_visual_text_groupkfold(
        ted_features, q_feats, o_feats, correct_idx, groups,
        condition_name="ted-vt",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psc-csv", required=True, type=Path)
    parser.add_argument("--ted-csv", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--checkpoint-key", default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--variant", choices=["visual_only", "visual_text", "both"],
                        default="both")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    psc_items = load_psc_data(args.psc_csv, args.video_root)
    ted_items = load_ted_data(args.ted_csv, args.video_root)
    print(f"Loaded {len(psc_items)} PSC items, {len(ted_items)} TED items",
          file=sys.stderr)

    # PSC features for the visual-only probe; TED-clip features for both probes
    all_items = psc_items + [{"video_file": t["video_file"],
                               "video_path": t["video_path"],
                               "timestamp": t["timestamp"]}
                              for t in ted_items]
    feature_cache = extract_or_load_features(
        all_items, args.checkpoint, args.checkpoint_key, args.cache)

    psc_feats = torch.stack(
        [feature_cache[(it["video_file"], it["timestamp"])]
         for it in psc_items])
    psc_labels = torch.tensor(
        [LABEL_TO_IDX[it["label"]] for it in psc_items],
        dtype=torch.long)

    ted_feats = torch.stack(
        [feature_cache[(it["video_file"], it["timestamp"])]
         for it in ted_items])

    results = {}
    if args.variant in ("visual_only", "both"):
        results["visual_only"] = run_visual_only(
            psc_feats, psc_labels, ted_feats, ted_items)
        print(f"\n  visual_only: {results['visual_only']['accuracy']*100:.1f}%")

    if args.variant in ("visual_text", "both"):
        results["visual_text"] = run_visual_text(ted_feats, ted_items)
        print(f"\n  visual_text: "
              f"{results['visual_text']['overall_acc']*100:.1f}%")

    print(f"\nTED Results")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  TED items: {len(ted_items)}")
    for k, v in results.items():
        acc = v.get("accuracy") or v.get("overall_acc")
        print(f"  {k:>14s}: {acc*100:.1f}%")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
