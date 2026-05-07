"""
sweep_psc_ccr_per_epoch.py — PSC + (optional) CCR Probe Sweep Across All Adaptation Epochs

Wraps train_psc_ccr_probes.py: imports its helpers, loops over the 7 saved
epoch checkpoints plus base, runs PSC at all 3 granularities for each.

Use this to decide which epoch should be the released Tacit checkpoint
based on actual benchmark probe accuracy rather than train loss alone.

Usage (from /home/tacit_experiment):
  python -u sweep_psc_ccr_per_epoch.py \
    --psc-csv /home/tacit_annotation/tacit_training_op_2026-05-02_with_clipid.csv \
    --video-root /home/downloads \
    2>&1 | grep --line-buffered -v "av1|Failed to get pixel|hardware accelerated|Get current frame" \
    | tee psc_ccr_per_epoch_results.log

Add --skip-ccr to run PSC only (faster, ~7 minutes per epoch).
Add --include-ccr to run CCR alongside (slower, ~20 minutes per epoch).
Default: PSC only.

Output:
  psc_ccr_per_epoch_results.json  — structured results, one entry per epoch
  /home/tacit_experiment/_features_cache/{condition}.pt  — per-condition feature caches
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
# Import everything we need from the existing probe script.
# This relies on train_psc_ccr_probes.py being in the same directory or on PYTHONPATH.
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, '/home/tacit_experiment')

from train_psc_ccr_probes import (
    DEVICE, WINDOW_SEC, N_FOLDS,
    MERGE_A, MERGE_B, MERGE_C,
    apply_merge,
    extract_frames_at_timestamp,
    build_encoder, load_checkpoint, encode_video,
    run_psc_groupkfold,
    construct_ccr_groups, run_ccr_leave_one_out,
    find_video_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("/home/tacit_experiment/adapted_v3_ema_motion_checkpoints")
BASE_CHECKPOINT = "/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"
FEATURE_CACHE_DIR = Path("/home/tacit_experiment/_features_cache")
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_conditions():
    """Build (name, ckpt_path, ckpt_key) tuples for base + all 7 epochs."""
    conds = [("base", BASE_CHECKPOINT, "ema_encoder")]
    for e in range(1, 8):
        ckpt = CHECKPOINT_DIR / f"adapted_v3_ema_motion_epoch{e:02d}.pth"
        conds.append((f"epoch_{e}", str(ckpt), "model_state"))
    return conds


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction with disk caching
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_for_condition(cond_name, ckpt_path, ckpt_key, raw_items, force=False):
    """Extract V-JEPA features for all items under one encoder condition.

    Caches the result to disk; on rerun reads cache unless force=True.
    Returns a dict: {(video_file, timestamp): feature_tensor (cpu)}.
    """
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
    features_cache = {}
    for i, item in enumerate(raw_items):
        key = (item['video_file'], item['timestamp'])
        if key not in features_cache:
            frames = extract_frames_at_timestamp(
                item['video_path'], item['timestamp'], window_sec=WINDOW_SEC)
            features_cache[key] = encode_video(encoder, frames)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(raw_items)} ({time.time()-t0:.0f}s)")

    print(f"  [{cond_name}] Extracted {len(features_cache)} unique features in {time.time()-t0:.0f}s")

    # Free encoder before saving cache
    del encoder
    torch.cuda.empty_cache()

    torch.save(features_cache, cache_path)
    print(f"  [{cond_name}] Cached features to {cache_path.name}")

    return features_cache


# ─────────────────────────────────────────────────────────────────────────────
# Run PSC + CCR for one condition given its feature cache
# ─────────────────────────────────────────────────────────────────────────────

def run_probes_for_condition(cond_name, features_cache, raw_items, video_to_group,
                              run_ccr=False):
    """Run PSC at 3 granularities and (optionally) CCR. Returns nested results dict."""
    cond_results = {}

    # PSC at 3 granularity levels
    merge_options = [
        ("23_class", MERGE_A),
        ("20_class", MERGE_B),
        ("10_class", MERGE_C),
    ]

    for merge_name, merge_map in merge_options:
        items = []
        for raw in raw_items:
            merged = apply_merge(raw['raw_label'], merge_map)
            if merged is None:
                continue
            items.append({**raw, 'merged_label': merged})

        class_names = sorted(set(item['merged_label'] for item in items))
        label_to_idx = {name: i for i, name in enumerate(class_names)}

        features = torch.stack([features_cache[(item['video_file'], item['timestamp'])]
                               for item in items])
        labels = torch.tensor([label_to_idx[item['merged_label']] for item in items],
                              dtype=torch.long)
        groups = np.array([video_to_group[item['video_file']] for item in items])

        result = run_psc_groupkfold(
            features, labels, groups, class_names, cond_name,
            n_folds=N_FOLDS, n_epochs=100, lr=1e-3)

        print(f"    [{cond_name}] PSC-{merge_name}: {result['overall_acc']:.3f} "
              f"(folds: {result['fold_accs']})")

        cond_results[f"psc_{merge_name}"] = result

    # CCR (optional, slower)
    if run_ccr:
        print(f"  [{cond_name}] Running CCR (leave-one-video-out, ~20 min per encoder)...")
        ccr_groups = construct_ccr_groups(raw_items)

        filtered_groups = []
        ccr_group_features = []
        for group in ccr_groups:
            clip_feats = []
            for item in group['items']:
                key = (item['video_file'], item['timestamp'])
                if key in features_cache:
                    clip_feats.append(features_cache[key])
            if len(clip_feats) >= 4:
                filtered_groups.append(group)
                ccr_group_features.append(clip_feats)

        if len(filtered_groups) >= 3:
            ccr_result = run_ccr_leave_one_out(ccr_group_features, filtered_groups, cond_name)
            cond_results['ccr'] = ccr_result
        else:
            print(f"    [{cond_name}] WARNING: only {len(filtered_groups)} CCR groups, skipping")

    return cond_results


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results, run_ccr):
    print(f"\n\n{'=' * 90}")
    print("SWEEP SUMMARY — PSC + CCR Across Adaptation Epochs")
    print(f"{'=' * 90}")

    # PSC headline table
    print(f"\n{'Condition':<12} {'PSC-23':>10} {'PSC-20':>10} {'PSC-10':>10}", end="")
    if run_ccr:
        print(f" {'CCR-tau':>10} {'CCR-pair':>10}", end="")
    print()
    print(f"{'-' * 90}")

    for cond_name, results in all_results.items():
        psc23 = results.get('psc_23_class', {}).get('overall_acc', None)
        psc20 = results.get('psc_20_class', {}).get('overall_acc', None)
        psc10 = results.get('psc_10_class', {}).get('overall_acc', None)
        ccr_tau = results.get('ccr', {}).get('mean_tau', None)
        ccr_pair = results.get('ccr', {}).get('mean_pairwise_acc', None)

        def fmt(v):
            return f"{v:>10.3f}" if v is not None else f"{'—':>10}"

        print(f"{cond_name:<12} {fmt(psc23)} {fmt(psc20)} {fmt(psc10)}", end="")
        if run_ccr:
            print(f" {fmt(ccr_tau)} {fmt(ccr_pair)}", end="")
        print()

    # Best-per-task callout
    print(f"\n{'-' * 90}")
    print("BEST EPOCH PER TASK")
    print(f"{'-' * 90}")
    for task in ['psc_23_class', 'psc_20_class', 'psc_10_class']:
        scores = {c: r.get(task, {}).get('overall_acc', -1) for c, r in all_results.items()}
        best_cond = max(scores, key=scores.get)
        print(f"  {task:<20} best at {best_cond:<12} ({scores[best_cond]:.3f})")
    if run_ccr:
        for task, label in [('mean_tau', 'CCR tau'), ('mean_pairwise_acc', 'CCR pair acc')]:
            scores = {c: r.get('ccr', {}).get(task, -2) for c, r in all_results.items()}
            best_cond = max(scores, key=scores.get)
            print(f"  {label:<20} best at {best_cond:<12} ({scores[best_cond]:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psc-csv", required=True)
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--include-ccr", action="store_true",
                        help="Also run CCR for each epoch (slower, ~20 min per condition)")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore cached features and recompute")
    parser.add_argument("--only-conditions", nargs="+", default=None,
                        help="Run only these conditions (e.g. 'base epoch_1 epoch_3'). "
                             "Default: all 8.")
    args = parser.parse_args()

    print("=" * 90)
    print("TACIT — PSC + CCR Probe Sweep Across Epochs")
    print("=" * 90)
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_SEC}s")
    print(f"Run CCR: {args.include_ccr}")
    print(f"Cache dir: {FEATURE_CACHE_DIR}")

    # Load annotations (same as parent script)
    raw_items = []
    with open(args.psc_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
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

    unique_videos = sorted(set(item['video_file'] for item in raw_items))
    video_to_group = {v: i for i, v in enumerate(unique_videos)}

    # Build full condition list, filter if --only-conditions
    all_conds = get_conditions()
    if args.only_conditions:
        all_conds = [c for c in all_conds if c[0] in args.only_conditions]
    print(f"Conditions to run: {[c[0] for c in all_conds]}")

    # Load existing partial results if any
    out_path = Path("/home/tacit_experiment/psc_ccr_per_epoch_results.json")
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
        print(f"\nLoaded existing partial results: {list(all_results.keys())}")
    else:
        all_results = {}

    # Run sweep
    for cond_name, ckpt_path, ckpt_key in all_conds:
        # Determine which tasks this condition still needs.
        # PSC always required; CCR only if --include-ccr.
        existing = all_results.get(cond_name, {})
        has_psc = all(f"psc_{g}" in existing for g in ("23_class", "20_class", "10_class"))
        has_ccr = "ccr" in existing
        needs_psc = not has_psc
        needs_ccr = args.include_ccr and not has_ccr

        if not (needs_psc or needs_ccr) and not args.force_recompute:
            print(f"\n[SKIP] {cond_name} already has all requested results.")
            continue

        print(f"\n{'=' * 90}")
        print(f"CONDITION: {cond_name}")
        print(f"  needs PSC: {needs_psc}, needs CCR: {needs_ccr}")
        print(f"{'=' * 90}")

        features_cache = extract_features_for_condition(
            cond_name, ckpt_path, ckpt_key, raw_items,
            force=args.force_recompute)

        if features_cache is None:
            continue

        # If PSC already done and we're only adding CCR, run CCR-only path.
        # Otherwise run the full probe pipeline.
        if needs_psc:
            new_results = run_probes_for_condition(
                cond_name, features_cache, raw_items, video_to_group,
                run_ccr=needs_ccr)
        else:
            # PSC already exists; only need CCR. Add it to the existing dict.
            new_results = dict(existing)  # copy existing PSC entries
            print(f"  [{cond_name}] Adding CCR (PSC already cached in JSON)")
            ccr_groups = construct_ccr_groups(raw_items)
            filtered_groups = []
            ccr_group_features = []
            for group in ccr_groups:
                clip_feats = []
                for item in group['items']:
                    key = (item['video_file'], item['timestamp'])
                    if key in features_cache:
                        clip_feats.append(features_cache[key])
                if len(clip_feats) >= 4:
                    filtered_groups.append(group)
                    ccr_group_features.append(clip_feats)
            if len(filtered_groups) >= 3:
                new_results['ccr'] = run_ccr_leave_one_out(
                    ccr_group_features, filtered_groups, cond_name)
            else:
                print(f"    [{cond_name}] WARNING: only {len(filtered_groups)} CCR groups")

        all_results[cond_name] = new_results

        # Persist after every condition so partial failures don't lose progress
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [{cond_name}] Saved partial results to {out_path}")

    print_summary(all_results, args.include_ccr)
    print(f"\nFinal results: {out_path}")
    print(f"\n{'=' * 90}")
    print("DONE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()