"""
sweep_ccr_only_per_epoch.py — From-scratch CCR sweep across all adaptation epochs

Purpose: independent CCR run with no shared state with prior sweeps. Re-extracts
features per condition and runs CCR fresh, writing to a separate output file.
This is for use when an earlier add-CCR-to-existing-results path failed and we
want to start clean.

Wraps train_psc_ccr_probes.py: imports its CCR helpers directly, no
re-implementation. Caches its own features at /home/tacit_experiment/_features_cache_ccr/
to avoid touching the PSC sweep's cache.

Usage (from /home/tacit_experiment):
  python -u sweep_ccr_only_per_epoch.py \
    --psc-csv /home/tacit_annotation/tacit_training_op_2026-05-02_with_clipid.csv \
    --video-root /home/downloads \
    2>&1 | grep --line-buffered -v "av1\\|Failed to get pixel\\|hardware accelerated\\|Get current frame" \
    | tee ccr_per_epoch_results.log

Output:
  /home/tacit_experiment/ccr_per_epoch_results.json
  /home/tacit_experiment/_features_cache_ccr/{condition}.pt  (per-condition feature caches)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Imports from existing probe script — no re-implementation
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, '/home/tacit_experiment')

from train_psc_ccr_probes import (
    DEVICE, WINDOW_SEC,
    extract_frames_at_timestamp,
    build_encoder, load_checkpoint, encode_video,
    construct_ccr_groups, run_ccr_leave_one_out,
    find_video_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("/home/tacit_experiment/adapted_v3_ema_motion_checkpoints")
BASE_CHECKPOINT = "/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"

# Separate cache dir from PSC sweep — we don't share state with that file
FEATURE_CACHE_DIR = Path("/home/tacit_experiment/_features_cache_ccr")
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Separate output JSON — we don't touch psc_ccr_per_epoch_results.json
OUTPUT_PATH = Path("/home/tacit_experiment/ccr_per_epoch_results.json")


def get_conditions():
    """(name, ckpt_path, ckpt_key) for base + all 7 epochs."""
    conds = [("base", BASE_CHECKPOINT, "ema_encoder")]
    for e in range(1, 8):
        ckpt = CHECKPOINT_DIR / f"adapted_v3_ema_motion_epoch{e:02d}.pth"
        conds.append((f"epoch_{e}", str(ckpt), "model_state"))
    return conds


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — fresh per condition
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_fresh(cond_name, ckpt_path, ckpt_key, raw_items, force=False):
    """Build {(video_file, timestamp): feature} for one condition. Disk-cached."""
    cache_path = FEATURE_CACHE_DIR / f"{cond_name}.pt"
    if cache_path.exists() and not force:
        print(f"  [{cond_name}] Loading cached features from {cache_path.name}")
        return torch.load(cache_path, map_location='cpu')

    if not Path(ckpt_path).exists():
        print(f"  [{cond_name}] SKIP: checkpoint not found at {ckpt_path}")
        return None

    print(f"  [{cond_name}] Loading encoder from {Path(ckpt_path).name}...")
    encoder = build_encoder()
    encoder = load_checkpoint(encoder, ckpt_path, ckpt_key)
    encoder = encoder.to(DEVICE).eval()

    print(f"  [{cond_name}] Extracting features for {len(raw_items)} items...")
    t0 = time.time()
    cache = {}
    for i, item in enumerate(raw_items):
        key = (item['video_file'], item['timestamp'])
        if key not in cache:
            frames = extract_frames_at_timestamp(
                item['video_path'], item['timestamp'], window_sec=WINDOW_SEC)
            cache[key] = encode_video(encoder, frames)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(raw_items)} ({time.time()-t0:.0f}s)")
    print(f"  [{cond_name}] Extracted {len(cache)} features in {time.time()-t0:.0f}s")

    del encoder
    torch.cuda.empty_cache()

    torch.save(cache, cache_path)
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# CCR runner per condition
# ─────────────────────────────────────────────────────────────────────────────

def run_ccr_for_condition(cond_name, features_cache, raw_items):
    """Run CCR leave-one-video-out for a single encoder condition."""
    print(f"\n  [{cond_name}] Constructing CCR groups...")
    ccr_groups = construct_ccr_groups(raw_items)
    print(f"  [{cond_name}] Initial CCR groups (≥4 clips, distinct states): {len(ccr_groups)}")

    filtered_groups = []
    ccr_group_features = []
    missing_count = 0
    for group in ccr_groups:
        clip_feats = []
        missing_in_group = 0
        for item in group['items']:
            key = (item['video_file'], item['timestamp'])
            if key in features_cache:
                clip_feats.append(features_cache[key])
            else:
                missing_in_group += 1
        if missing_in_group > 0:
            missing_count += missing_in_group
            print(f"    [warn] {group['video_file']}: {missing_in_group}/{len(group['items'])} clips missing from feature cache")
        if len(clip_feats) >= 4:
            filtered_groups.append(group)
            ccr_group_features.append(clip_feats)
        else:
            print(f"    [drop] {group['video_file']}: only {len(clip_feats)} clips with features (<4)")

    if missing_count > 0:
        print(f"  [{cond_name}] WARNING: {missing_count} clip features missing from cache across all groups")

    print(f"  [{cond_name}] CCR groups with ≥4 cached features: {len(filtered_groups)}")
    print(f"  [{cond_name}] Total pairs across groups: "
          f"{sum(g['n_clips'] * (g['n_clips']-1) // 2 for g in filtered_groups)}")

    if len(filtered_groups) < 3:
        print(f"  [{cond_name}] CRITICAL: only {len(filtered_groups)} groups have features, need ≥3 — skipping CCR")
        return None

    print(f"  [{cond_name}] Running CCR leave-one-video-out across {len(filtered_groups)} groups...")
    t0 = time.time()
    ccr_result = run_ccr_leave_one_out(ccr_group_features, filtered_groups, cond_name)
    print(f"  [{cond_name}] CCR done in {(time.time()-t0)/60:.1f} min")
    return ccr_result


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results):
    print(f"\n\n{'=' * 80}")
    print("CCR SWEEP SUMMARY — Across Adaptation Epochs")
    print(f"{'=' * 80}")

    print(f"\n{'Condition':<12} {'Tau':>10} {'Tau std':>10} {'Pair acc':>12} {'N groups':>10}")
    print("-" * 60)

    for cond, ccr in all_results.items():
        if ccr is None:
            print(f"{cond:<12}  (no result)")
            continue
        tau = ccr.get('mean_tau', None)
        tau_std = ccr.get('std_tau', None)
        pair = ccr.get('mean_pairwise_acc', None)
        n = ccr.get('n_groups', None)

        def fmt(v, w=10, prec=3):
            return f"{v:>{w}.{prec}f}" if v is not None else f"{'-':>{w}}"

        print(f"{cond:<12} {fmt(tau)} {fmt(tau_std)} {fmt(pair, 12)} {n if n is not None else '-':>10}")

    # Best by each metric
    print("\nBest epoch per metric:")
    for metric, label in [('mean_tau', "Kendall's tau"),
                          ('mean_pairwise_acc', 'Pairwise accuracy')]:
        scores = {c: r.get(metric, -float('inf'))
                  for c, r in all_results.items() if r is not None}
        if scores:
            best = max(scores, key=scores.get)
            print(f"  {label:<22} best at {best:<10} ({scores[best]:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psc-csv", required=True)
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--only-conditions", nargs="+", default=None,
                        help="Run only these conditions (e.g. 'base epoch_3')")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore existing feature caches and CCR results")
    args = parser.parse_args()

    print("=" * 80)
    print("TACIT — CCR Probe Sweep (from scratch, independent of PSC sweep)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_SEC}s")
    print(f"Feature cache dir: {FEATURE_CACHE_DIR}")
    print(f"Output JSON: {OUTPUT_PATH}")

    # Load annotations
    raw_items = []
    with open(args.psc_csv) as f:
        for row in csv.DictReader(f):
            video_path = find_video_path(row['video_file'], args.video_root)
            if video_path is None:
                continue
            raw_items.append({
                'clip_id': row['clip_id'],
                'video_file': row['video_file'],
                'video_path': video_path,
                'timestamp': float(row['timestamp_seconds']),
                'raw_label': row['physical_state'],
            })

    print(f"\nLoaded: {len(raw_items)} items from "
          f"{len(set(r['video_file'] for r in raw_items))} videos")

    # Build CCR groups once just to print expected count up front
    expected_groups = construct_ccr_groups(raw_items)
    print(f"Expected CCR groups (≥4 distinct-state clips per video): {len(expected_groups)}")

    # Conditions
    all_conds = get_conditions()
    if args.only_conditions:
        all_conds = [c for c in all_conds if c[0] in args.only_conditions]
    print(f"\nConditions to run: {[c[0] for c in all_conds]}")

    # Resume from existing partial results
    all_results = {}
    if OUTPUT_PATH.exists() and not args.force_recompute:
        with open(OUTPUT_PATH) as f:
            all_results = json.load(f)
        print(f"Resuming: existing CCR results for {list(all_results.keys())}")

    # Sweep
    for cond_name, ckpt_path, ckpt_key in all_conds:
        if cond_name in all_results and not args.force_recompute:
            print(f"\n[SKIP] {cond_name} already in CCR results.")
            continue

        print(f"\n{'=' * 80}")
        print(f"CONDITION: {cond_name}")
        print(f"{'=' * 80}")

        features_cache = extract_features_fresh(
            cond_name, ckpt_path, ckpt_key, raw_items, force=args.force_recompute)
        if features_cache is None:
            print(f"  [{cond_name}] No features available, skipping")
            continue

        ccr_result = run_ccr_for_condition(cond_name, features_cache, raw_items)
        all_results[cond_name] = ccr_result

        # Persist after every condition — protects against partial failures
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [{cond_name}] Saved partial results to {OUTPUT_PATH}")

    print_summary(all_results)

    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()