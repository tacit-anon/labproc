"""
sweep_ccr_same_state_per_epoch.py — Same-State CCR Sweep Across Adaptation Epochs

Exploratory sweep: same probe architecture as train_ccr_same_state.py
(concat-based PairwiseOrderProbe with BCE loss), wrapped to run across
base + 7 adaptation epochs.

This is exploratory work to see whether adaptation buys anything for Same-State
CCR with the current 23-group annotated dataset. Results from this sweep are
NOT intended for the v1 paper -- the existing v1 paper text marks Same-State
CCR as "in progress" and that framing remains.

Usage (from /home/tacit_experiment):
  python -u sweep_ccr_same_state_per_epoch.py \
    --groups-csv /home/tacit_annotation/ccr_same_state.csv \
    --video-root /home/downloads \
    2>&1 | grep --line-buffered -v "av1\\|Failed to get pixel\\|hardware accelerated\\|Get current frame" \
    | tee ccr_same_state_per_epoch.log

Optional flags:
  --only-conditions base epoch_4   # restrict conditions
  --force-recompute                # ignore caches and existing results

Output:
  /home/tacit_experiment/ccr_same_state_per_epoch_results.json
  /home/tacit_experiment/_features_cache_ccr_ss/{condition}.pt
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Imports from existing probe script
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, '/home/tacit_experiment')

from train_ccr_same_state import (
    DEVICE, WINDOW_SEC,
    extract_frames_at_timestamp,
    build_encoder, load_checkpoint, encode_video,
    evaluate_leave_one_out,
    find_video_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("/home/tacit_experiment/adapted_v3_ema_motion_checkpoints")
BASE_CHECKPOINT = "/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"

# Separate cache directory — this is exploratory, don't touch standard CCR caches
FEATURE_CACHE_DIR = Path("/home/tacit_experiment/_features_cache_ccr_ss")
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = Path("/home/tacit_experiment/ccr_same_state_per_epoch_results.json")


def get_conditions():
    """(name, ckpt_path, ckpt_key) for base + all 7 epochs."""
    conds = [("base", BASE_CHECKPOINT, "ema_encoder")]
    for e in range(1, 8):
        ckpt = CHECKPOINT_DIR / f"adapted_v3_ema_motion_epoch{e:02d}.pth"
        conds.append((f"epoch_{e}", str(ckpt), "model_state"))
    return conds


# ─────────────────────────────────────────────────────────────────────────────
# Load Same-State CCR groups (mirrors the loader in train_ccr_same_state.main())
# ─────────────────────────────────────────────────────────────────────────────

def load_same_state_groups(groups_csv, video_root):
    """Load groups from the Same-State CCR annotation CSV."""
    groups = []
    with open(groups_csv) as f:
        for row in csv.DictReader(f):
            video_path = find_video_path(
                row['video_file'], row.get('video_path'), video_root)
            if video_path is None:
                continue

            timestamps = []
            for j in range(1, 6):
                ts_val = row.get(f'ts_{j}')
                if ts_val and ts_val != '':
                    timestamps.append(float(ts_val))

            if len(timestamps) < 3:
                continue

            groups.append({
                'group_id': row['group_id'],
                'video_file': row['video_file'],
                'video_path': video_path,
                'state': row['state_label'],
                'timestamps': sorted(timestamps),
            })
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction with disk caching
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_for_condition(cond_name, ckpt_path, ckpt_key, groups,
                                     force=False):
    """Build per-group feature lists for one V-JEPA condition. Disk-cached."""
    cache_path = FEATURE_CACHE_DIR / f"{cond_name}.pt"

    if cache_path.exists() and not force:
        print(f"  [{cond_name}] Loading cached features from {cache_path.name}")
        cached = torch.load(cache_path, map_location='cpu')
        # Validate: should be a list of (group_id, list-of-feats)
        # Reconstruct in current group order
        feats_by_id = {entry['group_id']: entry['features'] for entry in cached}
        group_features = []
        group_states = []
        missing = []
        for group in groups:
            if group['group_id'] in feats_by_id:
                group_features.append(feats_by_id[group['group_id']])
                group_states.append(group['state'])
            else:
                missing.append(group['group_id'])
        if missing:
            print(f"  [{cond_name}] WARN: cached features missing for "
                  f"{len(missing)} group_ids; falling back to fresh extraction")
        else:
            return group_features, group_states

    if not Path(ckpt_path).exists():
        print(f"  [{cond_name}] SKIP: checkpoint not found at {ckpt_path}")
        return None, None

    print(f"  [{cond_name}] Loading encoder from {Path(ckpt_path).name}...")
    encoder = build_encoder()
    encoder = load_checkpoint(encoder, ckpt_path, ckpt_key)
    encoder = encoder.to(DEVICE).eval()

    total_clips = sum(len(g['timestamps']) for g in groups)
    print(f"  [{cond_name}] Extracting features for {total_clips} clips "
          f"across {len(groups)} groups...")

    t0 = time.time()
    group_features = []
    group_states = []
    cache_records = []
    for gi, group in enumerate(groups):
        clip_feats = []
        for ts in group['timestamps']:
            frames = extract_frames_at_timestamp(group['video_path'], ts)
            feat = encode_video(encoder, frames)
            clip_feats.append(feat)
        group_features.append(clip_feats)
        group_states.append(group['state'])
        cache_records.append({
            'group_id': group['group_id'],
            'video_file': group['video_file'],
            'state': group['state'],
            'features': clip_feats,
        })
        if (gi + 1) % 5 == 0:
            print(f"    Group {gi+1}/{len(groups)} ({time.time()-t0:.0f}s)")
    print(f"  [{cond_name}] Extracted features for {total_clips} clips "
          f"in {time.time()-t0:.0f}s")

    del encoder
    torch.cuda.empty_cache()

    torch.save(cache_records, cache_path)
    return group_features, group_states


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results):
    print(f"\n\n{'=' * 80}")
    print("SAME-STATE CCR — Per-Epoch Sweep Summary")
    print(f"{'=' * 80}")
    print(f"\n  Random baselines: tau=0.000, pair_acc=0.500\n")

    if not all_results:
        print("(no results)")
        return

    print(f"{'Condition':<12} {'Mean tau (mean+-std)':>22} {'Pair acc (mean+-std)':>22} {'N groups':>10}")
    print("-" * 70)

    for cond, r in all_results.items():
        if r is None:
            print(f"{cond:<12}  (no result)")
            continue
        tau_str = f"{r['mean_tau']:+.3f}+-{r['std_tau']:.3f}"
        pair_str = f"{r['mean_pair_acc']:.3f}+-{r['std_pair_acc']:.3f}"
        n_groups = len(r.get('per_group', []))
        print(f"{cond:<12}  {tau_str:>22}  {pair_str:>22}  {n_groups:>10}")

    print("\nBest epoch per metric:")
    for metric, label in [('mean_tau', "Kendall's tau"),
                          ('mean_pair_acc', 'Pair acc')]:
        scores = {c: r.get(metric, -float('inf'))
                  for c, r in all_results.items() if r is not None}
        if scores:
            best = max(scores, key=scores.get)
            print(f"  {label:<22} best at {best:<10} ({scores[best]:+.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups-csv", required=True,
                        help="Same-State CCR annotation CSV")
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--n-epochs", type=int, default=100,
                        help="Probe training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--only-conditions", nargs="+", default=None,
                        help="Run only these conditions (e.g. 'base epoch_4')")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore caches and existing results")
    args = parser.parse_args()

    print("=" * 80)
    print("TACIT — Same-State CCR Per-Epoch Sweep (exploratory)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_SEC}s")
    print(f"Probe: concat-based PairwiseOrderProbe (matches train_ccr_same_state.py)")
    print(f"Feature cache dir: {FEATURE_CACHE_DIR}")
    print(f"Output JSON: {OUTPUT_PATH}")

    # Load groups
    groups = load_same_state_groups(args.groups_csv, args.video_root)
    print(f"\nLoaded {len(groups)} Same-State CCR groups with valid videos")

    state_counts = defaultdict(int)
    for g in groups:
        state_counts[g['state']] += 1
    print(f"State distribution:")
    for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
        print(f"  {state}: {count}")

    if len(groups) < 3:
        print("\nERROR: Need at least 3 groups for leave-one-out evaluation")
        return

    # Conditions
    all_conds = get_conditions()
    if args.only_conditions:
        all_conds = [c for c in all_conds if c[0] in args.only_conditions]
    print(f"\nConditions to run: {[c[0] for c in all_conds]}")

    # Resume from existing results
    all_results = {}
    if OUTPUT_PATH.exists() and not args.force_recompute:
        with open(OUTPUT_PATH) as f:
            all_results = json.load(f)
        print(f"\nResuming: existing results for {list(all_results.keys())}")

    # Sweep
    for cond_name, ckpt_path, ckpt_key in all_conds:
        if cond_name in all_results and not args.force_recompute:
            print(f"\n[SKIP] {cond_name} already in results.")
            continue

        print(f"\n{'=' * 80}")
        print(f"CONDITION: {cond_name}")
        print(f"{'=' * 80}")

        group_features, group_states = extract_features_for_condition(
            cond_name, ckpt_path, ckpt_key, groups,
            force=args.force_recompute)

        if group_features is None:
            print(f"  [{cond_name}] Skipping — no features available")
            continue

        print(f"\n  [{cond_name}] Running leave-one-group-out evaluation "
              f"across {len(group_features)} groups...")
        result = evaluate_leave_one_out(
            group_features, group_states, cond_name,
            n_epochs=args.n_epochs, lr=args.lr)

        print(f"\n  [{cond_name}] Mean tau:      {result['mean_tau']:+.3f} +- {result['std_tau']:.3f}")
        print(f"  [{cond_name}] Mean pair acc: {result['mean_pair_acc']:.3f} +- {result['std_pair_acc']:.3f}")

        all_results[cond_name] = result

        # Persist after every condition
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [{cond_name}] Saved partial results to {OUTPUT_PATH}")

    print_summary(all_results)
    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")
    print("\nReminder: this is exploratory data on the current 23-group dataset.")
    print("Results are NOT folded into the v1 paper unless explicitly decided.")


if __name__ == "__main__":
    main()