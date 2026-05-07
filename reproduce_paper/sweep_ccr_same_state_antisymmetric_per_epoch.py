"""
sweep_ccr_same_state_antisymmetric_per_epoch.py
---------------------------------------------
Antisymmetric probe sweep for Same-State CCR.

Reuses cached features from sweep_ccr_same_state_per_epoch.py
(/home/tacit_experiment/_features_cache_ccr_ss/{cond}.pt) and the antisymmetric
probe machinery from sweep_ccr_antisymmetric_per_epoch.py (AntisymmetricCCRProbe,
margin_loss, build_pairs_from_groups, evaluate_held_out_group, train_one_seed).

Writes results to ccr_same_state_antisym_per_epoch_results.json -- separate
file from the concat-probe sweep so both can be compared.

Usage (from /home/tacit_experiment, after the concat-probe sweep has populated
the feature cache):

  python -u sweep_ccr_same_state_antisymmetric_per_epoch.py \
    --groups-csv ccr_same_state_v2.csv \
    --video-root /home/downloads \
    2>&1 | tee ccr_ss_antisym_per_epoch.log

Expected runtime: ~5-10 min total (probe-only, all on cached features).
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
# Imports
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, '/home/tacit_experiment')

# Same-State CCR data loader (group construction, video resolution)
from sweep_ccr_same_state_per_epoch import load_same_state_groups

# Antisymmetric probe + LOGO machinery
from sweep_ccr_antisymmetric_per_epoch import (
    DEVICE,
    build_pairs_from_groups,
    evaluate_held_out_group,
    train_one_seed,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("/home/tacit_experiment/adapted_v3_ema_motion_checkpoints")
BASE_CHECKPOINT = "/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"

# Reuse the cache built by the concat-probe Same-State sweep
FEATURE_CACHE_DIR = Path("/home/tacit_experiment/_features_cache_ccr_ss")

# Separate output JSON to preserve concat-probe results for comparison
OUTPUT_PATH = Path("/home/tacit_experiment/ccr_same_state_antisym_per_epoch_results.json")


def get_conditions():
    """(name, ckpt_path, ckpt_key) for base + all 7 epochs."""
    conds = [("base", BASE_CHECKPOINT, "ema_encoder")]
    for e in range(1, 8):
        ckpt = CHECKPOINT_DIR / f"adapted_v3_ema_motion_epoch{e:02d}.pth"
        conds.append((f"epoch_{e}", str(ckpt), "model_state"))
    return conds


# ─────────────────────────────────────────────────────────────────────────────
# Load cached features in the format the antisymmetric probe expects
# ─────────────────────────────────────────────────────────────────────────────

def load_group_features(cond_name, groups):
    """
    Load cached features and return them in the format expected by
    build_pairs_from_groups: a list-of-lists where outer list iterates over
    groups and inner list iterates over clips in temporal order.

    The concat-probe Same-State sweep cached features as a list of records:
    [{'group_id': ..., 'video_file': ..., 'state': ..., 'features': [feat, ...]}, ...]

    We reorder them to match the order in 'groups' (which is the canonical order
    for the v2 CSV) and discard any cached entries that don't appear in groups.
    """
    cache_path = FEATURE_CACHE_DIR / f"{cond_name}.pt"
    if not cache_path.exists():
        return None, None

    cached = torch.load(cache_path, map_location='cpu')
    feats_by_id = {r['group_id']: r['features'] for r in cached}

    group_features = []
    group_states = []
    missing = []
    for g in groups:
        if g['group_id'] in feats_by_id:
            group_features.append(feats_by_id[g['group_id']])
            group_states.append(g['state'])
        else:
            missing.append(g['group_id'])

    if missing:
        print(f"  [{cond_name}] WARN: {len(missing)} groups missing from cache; "
              f"this sweep needs the concat-probe sweep to have populated them first")
        if len(missing) > len(groups) // 2:
            return None, None  # too many missing, abort this condition

    return group_features, group_states


# ─────────────────────────────────────────────────────────────────────────────
# Run antisymmetric probe across LOGO folds for one condition
# ─────────────────────────────────────────────────────────────────────────────

def run_antisymmetric_for_condition(cond_name, group_features, group_states,
                                      n_seeds=3, margin=0.5):
    """
    Leave-one-group-out evaluation with antisymmetric probe.
    Returns dict with mean/std tau, mean/std pair_acc, per-group results.
    """
    n_groups = len(group_features)
    if n_groups < 3:
        print(f"  [{cond_name}] CRITICAL: only {n_groups} groups, need >=3")
        return None

    embed_dim = group_features[0][0].shape[0]
    print(f"  [{cond_name}] Running LOGO across {n_groups} groups, "
          f"{n_seeds} seeds per fold, embed_dim={embed_dim}, margin={margin}")

    per_group_taus = []
    per_group_pair_accs = []
    total_pair_correct = 0
    total_pair_total = 0
    per_group_records = []

    t0 = time.time()
    for held_out_idx in range(n_groups):
        train_indices = [i for i in range(n_groups) if i != held_out_idx]

        train_a, train_b, train_target = build_pairs_from_groups(
            train_indices, [None] * n_groups, group_features)

        seed_taus = []
        seed_pair_accs = []
        seed_pair_correct = 0
        seed_pair_total = 0

        for seed in range(n_seeds):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)

            probe = train_one_seed(train_a, train_b, train_target,
                                    embed_dim, n_epochs=200, lr=1e-3, margin=margin)
            held_out_feats = group_features[held_out_idx]
            n_clips = len(held_out_feats)

            _, pair_correct, pair_total, tau = evaluate_held_out_group(
                probe, held_out_feats, n_clips)

            seed_taus.append(tau)
            seed_pair_accs.append(pair_correct / pair_total if pair_total else 0.0)
            seed_pair_correct += pair_correct
            seed_pair_total += pair_total

            del probe
            torch.cuda.empty_cache()

        per_group_taus.append(float(np.mean(seed_taus)))
        per_group_pair_accs.append(float(np.mean(seed_pair_accs)))
        total_pair_correct += seed_pair_correct
        total_pair_total += seed_pair_total

        per_group_records.append({
            'group_idx': held_out_idx,
            'state': group_states[held_out_idx],
            'n_clips': len(group_features[held_out_idx]),
            'tau_mean': round(float(np.mean(seed_taus)), 3),
            'tau_std': round(float(np.std(seed_taus)), 3),
            'pair_acc_mean': round(float(np.mean(seed_pair_accs)), 3),
            'pair_acc_std': round(float(np.std(seed_pair_accs)), 3),
        })

        print(f"    Group {held_out_idx + 1}/{n_groups} ({group_states[held_out_idx]}): "
              f"tau={np.mean(seed_taus):+.3f}+-{np.std(seed_taus):.3f}, "
              f"pair_acc={np.mean(seed_pair_accs):.3f}")

    macro_tau_mean = float(np.mean(per_group_taus))
    macro_tau_std = float(np.std(per_group_taus))
    macro_pair_mean = float(np.mean(per_group_pair_accs))
    macro_pair_std = float(np.std(per_group_pair_accs))
    micro_pair_acc = total_pair_correct / total_pair_total if total_pair_total else 0.0

    elapsed = time.time() - t0
    print(f"  [{cond_name}] DONE in {elapsed/60:.1f} min  "
          f"tau={macro_tau_mean:+.3f}+-{macro_tau_std:.3f}, "
          f"pair_acc(macro)={macro_pair_mean:.3f}+-{macro_pair_std:.3f}, "
          f"pair_acc(micro)={micro_pair_acc:.3f}")

    return {
        'mean_tau': round(macro_tau_mean, 3),
        'std_tau': round(macro_tau_std, 3),
        'mean_pair_acc': round(macro_pair_mean, 3),
        'std_pair_acc': round(macro_pair_std, 3),
        'micro_pair_acc': round(micro_pair_acc, 3),
        'n_groups': n_groups,
        'n_seeds': n_seeds,
        'margin': margin,
        'per_group': per_group_records,
        'probe_type': 'antisymmetric',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results, concat_results=None):
    """Print headline summary; optionally compare against concat-probe results."""
    print(f"\n\n{'=' * 95}")
    print("SAME-STATE CCR — Antisymmetric Probe Sweep Summary")
    print(f"{'=' * 95}")
    print(f"\n  Random baselines: tau=0.000, pair_acc=0.500\n")

    print(f"{'Condition':<12} {'Antisym tau':>20} {'Antisym pair':>20}", end="")
    if concat_results:
        print(f" {'Concat tau':>14} {'Concat pair':>14}", end="")
    print()
    print("-" * 95)

    for cond, r in all_results.items():
        if r is None:
            print(f"{cond:<12}  (no result)")
            continue
        tau_str = f"{r['mean_tau']:+.3f}+-{r['std_tau']:.3f}"
        pair_str = f"{r['mean_pair_acc']:.3f}+-{r['std_pair_acc']:.3f}"
        line = f"{cond:<12} {tau_str:>20} {pair_str:>20}"
        if concat_results and cond in concat_results and concat_results[cond] is not None:
            cr = concat_results[cond]
            ctau = cr.get('mean_tau', None)
            cpair = cr.get('mean_pair_acc', None)
            line += f"   {f'{ctau:+.3f}' if ctau is not None else '-':>10}"
            line += f"     {f'{cpair:.3f}' if cpair is not None else '-':>10}"
        print(line)

    print("\nBest epoch per metric (antisymmetric):")
    for metric, label in [('mean_tau', "Kendall's tau"),
                          ('mean_pair_acc', 'Pair accuracy (macro)')]:
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
                        help="Same-State CCR annotation CSV (v2 preferred)")
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--only-conditions", nargs="+", default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--compare-concat", action="store_true",
                        help="Print side-by-side comparison with concat-probe results")
    args = parser.parse_args()

    print("=" * 80)
    print("TACIT — Same-State CCR Antisymmetric Probe Sweep")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Margin: {args.margin}")
    print(f"Seeds per fold: {args.n_seeds}")
    print(f"Feature cache dir: {FEATURE_CACHE_DIR}")
    print(f"Output JSON: {OUTPUT_PATH}")

    # Load groups
    groups = load_same_state_groups(args.groups_csv, args.video_root)
    print(f"\nLoaded {len(groups)} Same-State CCR groups with valid videos")

    state_counts = defaultdict(int)
    for g in groups:
        state_counts[g['state']] += 1
    print("State distribution:")
    for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
        print(f"  {state}: {count}")

    if len(groups) < 3:
        print("\nERROR: Need at least 3 groups for leave-one-out evaluation")
        return

    # Verify the cache exists
    if not FEATURE_CACHE_DIR.exists() or not list(FEATURE_CACHE_DIR.glob("*.pt")):
        print(f"\nERROR: No cached features at {FEATURE_CACHE_DIR}")
        print("Run sweep_ccr_same_state_per_epoch.py first to populate the cache.")
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

        group_features, group_states = load_group_features(cond_name, groups)
        if group_features is None:
            print(f"  [{cond_name}] No usable features; skipping")
            continue

        result = run_antisymmetric_for_condition(
            cond_name, group_features, group_states,
            n_seeds=args.n_seeds, margin=args.margin)

        all_results[cond_name] = result

        with open(OUTPUT_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [{cond_name}] Saved to {OUTPUT_PATH}")

    # Optionally compare with concat-probe results
    concat_results = None
    if args.compare_concat:
        concat_path = Path("/home/tacit_experiment/ccr_same_state_per_epoch_results.json")
        if concat_path.exists():
            with open(concat_path) as f:
                concat_results = json.load(f)
            print(f"\nLoaded concat-probe results from {concat_path.name} for comparison")

    print_summary(all_results, concat_results=concat_results)
    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()