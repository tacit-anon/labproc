"""
sweep_ccr_antisymmetric_per_epoch.py — CCR sweep with antisymmetric probe

Reuses the cached features from sweep_ccr_only_per_epoch.py
(/home/tacit_experiment/_features_cache_ccr/{cond}.pt). Replaces the existing
concat-based pairwise probe with an antisymmetric probe:

    score(A, B) = g(A) - g(B)

where g is a single shared projection head (1024 -> 256 -> 1). This enforces
score(A, B) = -score(B, A) by construction, eliminating position-specific
shortcuts and halving parameter count.

Loss is triplet margin instead of BCE:
    loss = max(0, margin - target_sign * score)
where target_sign in {-1, +1} from the binary label (A before B = +1).

Output goes to a separate JSON
(/home/tacit_experiment/ccr_antisymmetric_per_epoch_results.json) so the
original concat-based results in ccr_per_epoch_results.json are preserved
for comparison.

Multi-seed: each fold runs 3 seeds, mean +/- std reported.

Usage (from /home/tacit_experiment):
  python -u sweep_ccr_antisymmetric_per_epoch.py \
    --psc-csv /home/tacit_annotation/tacit_training_op_2026-05-02_with_clipid.csv \
    --video-root /home/downloads \
    2>&1 | tee ccr_antisym_per_epoch.log

Optional flags:
  --only-conditions base epoch_4  # restrict to specific conditions
  --margin 0.5                    # triplet margin (default 0.5)
  --n-seeds 3                     # seeds per fold (default 3)
  --force-recompute               # ignore existing JSON, recompute all
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
import torch.nn as nn
from scipy.stats import kendalltau

# ---------------------------------------------------------------------
# Imports from existing probe script -- CCR group construction only
# ---------------------------------------------------------------------

sys.path.insert(0, '/home/tacit_experiment')

from train_psc_ccr_probes import (
    DEVICE, WINDOW_SEC,
    construct_ccr_groups,
    find_video_path,
)


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

CHECKPOINT_DIR = Path("/home/tacit_experiment/adapted_v3_ema_motion_checkpoints")
BASE_CHECKPOINT = "/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"

# Reuse cached features from the from-scratch CCR sweep
FEATURE_CACHE_DIR = Path("/home/tacit_experiment/_features_cache_ccr")

# Separate output to keep concat-based results intact for comparison
OUTPUT_PATH = Path("/home/tacit_experiment/ccr_antisymmetric_per_epoch_results.json")


def get_conditions():
    """(name, ckpt_path, ckpt_key) for base + all 7 epochs."""
    conds = [("base", BASE_CHECKPOINT, "ema_encoder")]
    for e in range(1, 8):
        ckpt = CHECKPOINT_DIR / f"adapted_v3_ema_motion_epoch{e:02d}.pth"
        conds.append((f"epoch_{e}", str(ckpt), "model_state"))
    return conds


# ---------------------------------------------------------------------
# Antisymmetric probe
# ---------------------------------------------------------------------

class AntisymmetricCCRProbe(nn.Module):
    """score(A, B) = g(A) - g(B), where g: 1024 -> 256 -> 1.

    By construction, score(A, B) = -score(B, A): no position shortcut possible.
    Trained with triplet margin loss, not BCE.
    """
    def __init__(self, embed_dim, hidden_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat_a, feat_b):
        # feat_a, feat_b: [B, embed_dim]
        # returns [B] scalar score; sign indicates A-before-B (positive)
        score_a = self.head(feat_a).squeeze(-1)
        score_b = self.head(feat_b).squeeze(-1)
        return score_a - score_b


def margin_loss(score, target_sign, margin=0.5):
    """target_sign in {-1, +1}; loss = max(0, margin - target_sign * score)."""
    return torch.clamp(margin - target_sign * score, min=0).mean()


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def build_pairs_from_groups(group_indices, ccr_groups, ccr_features):
    """Build all within-group pairs across the listed groups.

    Returns:
      pair_a, pair_b: tensors [N_pairs, embed_dim]
      target_sign:    tensor [N_pairs], -1 or +1
                      +1 means clip A came BEFORE clip B in true ordering.
    """
    pair_a, pair_b, target_sign = [], [], []
    for gi in group_indices:
        feats = ccr_features[gi]                 # list of N feature tensors in temporal order
        n = len(feats)
        for i in range(n):
            for j in range(i + 1, n):
                # i < j in true order, so target = +1 for ordering (i, j)
                pair_a.append(feats[i])
                pair_b.append(feats[j])
                target_sign.append(+1.0)
                # And include the reversed pair for symmetric training (target = -1)
                pair_a.append(feats[j])
                pair_b.append(feats[i])
                target_sign.append(-1.0)

    return (torch.stack(pair_a),
            torch.stack(pair_b),
            torch.tensor(target_sign, dtype=torch.float32))


def evaluate_held_out_group(probe, group_features, n_clips):
    """Predict pairwise ordering for one held-out group.

    Returns:
      pred_order:   list of clip indices, predicted temporal order (length n_clips)
      pair_correct: int count of correctly-ordered pairs
      pair_total:   int count of total pairs (= n*(n-1)/2)
      tau:          kendalltau between predicted and true order
    """
    probe.eval()
    with torch.no_grad():
        # Score each pair (i, j) with i < j (true order says i before j)
        feats = torch.stack(group_features).to(DEVICE)         # [n, embed_dim]
        # All pairs; evaluate score(i, j) for every i,j
        n = n_clips
        score_matrix = torch.zeros(n, n, device=DEVICE)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                score_matrix[i, j] = probe(feats[i:i+1], feats[j:j+1])

        # Pairwise accuracy: for true (i < j), check score(i, j) > 0
        pair_correct = 0
        pair_total = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair_total += 1
                if score_matrix[i, j].item() > 0:
                    pair_correct += 1

        # Predicted ordering: rank each clip by the number of clips it precedes
        # (sum of positive scores against all other clips)
        precedence_score = score_matrix.sum(dim=1).cpu().numpy()  # [n]
        # Sort descending: highest precedence_score = earliest in time
        pred_order = np.argsort(-precedence_score).tolist()
        true_order = list(range(n))
        tau, _ = kendalltau(true_order, pred_order)
        if np.isnan(tau):
            tau = 0.0

    return pred_order, pair_correct, pair_total, float(tau)


def train_one_seed(train_pairs_a, train_pairs_b, train_target,
                    embed_dim, n_epochs=200, lr=1e-3, margin=0.5):
    """Train the antisymmetric probe for one seed."""
    probe = AntisymmetricCCRProbe(embed_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)

    train_pairs_a = train_pairs_a.to(DEVICE)
    train_pairs_b = train_pairs_b.to(DEVICE)
    train_target = train_target.to(DEVICE)

    probe.train()
    for epoch in range(n_epochs):
        score = probe(train_pairs_a, train_pairs_b)
        loss = margin_loss(score, train_target, margin=margin)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return probe


def run_ccr_for_condition(cond_name, features_cache, ccr_groups,
                            n_seeds=3, margin=0.5):
    """Leave-one-group-out CCR with antisymmetric probe and multi-seed averaging."""

    # Build per-group ordered feature lists
    ccr_group_features = []
    for group in ccr_groups:
        clip_feats = []
        for item in group['items']:
            key = (item['video_file'], item['timestamp'])
            if key in features_cache:
                clip_feats.append(features_cache[key])
        if len(clip_feats) >= 4:
            ccr_group_features.append(clip_feats)
        else:
            print(f"  [{cond_name}] WARN: group {group['video_file']} has <4 cached features, skipping")

    n_groups = len(ccr_group_features)
    if n_groups < 3:
        print(f"  [{cond_name}] CRITICAL: only {n_groups} groups, need >=3")
        return None

    embed_dim = ccr_group_features[0][0].shape[0]
    print(f"  [{cond_name}] Running LOGO across {n_groups} groups, "
          f"{n_seeds} seeds per fold, embed_dim={embed_dim}")

    # Per-group results, averaged across seeds
    per_group_taus = []
    per_group_pair_accs = []
    total_pair_correct = 0
    total_pair_total = 0

    t0 = time.time()
    for held_out_idx in range(n_groups):
        train_indices = [i for i in range(n_groups) if i != held_out_idx]

        # Build training pairs from all other groups
        train_a, train_b, train_target = build_pairs_from_groups(
            train_indices, [None] * n_groups, ccr_group_features)

        # Multi-seed averaging on this fold
        seed_taus = []
        seed_pair_accs = []
        seed_pair_correct = 0
        seed_pair_total = 0

        for seed in range(n_seeds):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)

            probe = train_one_seed(train_a, train_b, train_target,
                                    embed_dim, n_epochs=200, lr=1e-3, margin=margin)
            held_out_feats = ccr_group_features[held_out_idx]
            n_clips = len(held_out_feats)

            _, pair_correct, pair_total, tau = evaluate_held_out_group(
                probe, held_out_feats, n_clips)

            seed_taus.append(tau)
            seed_pair_accs.append(pair_correct / pair_total if pair_total else 0.0)
            seed_pair_correct += pair_correct
            seed_pair_total += pair_total

            del probe
            torch.cuda.empty_cache()

        # Mean across seeds for this group
        per_group_taus.append({
            'mean': float(np.mean(seed_taus)),
            'std': float(np.std(seed_taus)),
            'per_seed': seed_taus,
        })
        per_group_pair_accs.append({
            'mean': float(np.mean(seed_pair_accs)),
            'std': float(np.std(seed_pair_accs)),
            'per_seed': seed_pair_accs,
        })

        # Aggregate over all seeds (for micro pair acc)
        total_pair_correct += seed_pair_correct
        total_pair_total += seed_pair_total

        print(f"    Group {held_out_idx + 1}/{n_groups}: "
              f"tau={np.mean(seed_taus):+.3f}+-{np.std(seed_taus):.3f}, "
              f"pair_acc={np.mean(seed_pair_accs):.3f}")

    # Aggregate metrics
    macro_tau_mean = float(np.mean([g['mean'] for g in per_group_taus]))
    macro_tau_std = float(np.std([g['mean'] for g in per_group_taus]))
    macro_pair_mean = float(np.mean([g['mean'] for g in per_group_pair_accs]))
    macro_pair_std = float(np.std([g['mean'] for g in per_group_pair_accs]))
    micro_pair_acc = total_pair_correct / total_pair_total if total_pair_total else 0.0

    elapsed = time.time() - t0
    print(f"  [{cond_name}] DONE in {elapsed/60:.1f} min  "
          f"tau={macro_tau_mean:+.3f}+-{macro_tau_std:.3f}, "
          f"pair_acc(macro)={macro_pair_mean:.3f}+-{macro_pair_std:.3f}, "
          f"pair_acc(micro)={micro_pair_acc:.3f}")

    return {
        'mean_tau': round(macro_tau_mean, 3),
        'std_tau': round(macro_tau_std, 3),
        'mean_pairwise_acc': round(macro_pair_mean, 3),
        'std_pairwise_acc': round(macro_pair_std, 3),
        'micro_pairwise_acc': round(micro_pair_acc, 3),
        'n_groups': n_groups,
        'n_seeds': n_seeds,
        'margin': margin,
        'per_group_tau': per_group_taus,
        'per_group_pair_acc': per_group_pair_accs,
        'probe_type': 'antisymmetric',
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_summary(all_results):
    print(f"\n\n{'=' * 90}")
    print("CCR ANTISYMMETRIC PROBE SWEEP — Summary")
    print(f"{'=' * 90}")
    print(f"\n{'Condition':<12} {'Tau (mean+-std)':>22} {'Pair acc (macro)':>22} {'Pair acc (micro)':>22}")
    print("-" * 90)
    for cond, r in all_results.items():
        if r is None:
            print(f"{cond:<12}  (no result)")
            continue
        tau_str = f"{r['mean_tau']:+.3f}+-{r['std_tau']:.3f}"
        pair_macro_str = f"{r['mean_pairwise_acc']:.3f}+-{r['std_pairwise_acc']:.3f}"
        pair_micro_str = f"{r['micro_pairwise_acc']:.3f}"
        print(f"{cond:<12}  {tau_str:>22}  {pair_macro_str:>22}  {pair_micro_str:>22}")

    print("\nBest epoch per metric:")
    for metric, label in [('mean_tau', "Kendall's tau"),
                          ('mean_pairwise_acc', 'Pair acc (macro)'),
                          ('micro_pairwise_acc', 'Pair acc (micro)')]:
        scores = {c: r.get(metric, -float('inf'))
                  for c, r in all_results.items() if r is not None}
        if scores:
            best = max(scores, key=scores.get)
            print(f"  {label:<22} best at {best:<10} ({scores[best]:.3f})")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psc-csv", required=True)
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--only-conditions", nargs="+", default=None)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("TACIT — CCR Antisymmetric Probe Sweep")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Margin: {args.margin}")
    print(f"Seeds per fold: {args.n_seeds}")
    print(f"Feature cache dir: {FEATURE_CACHE_DIR}")
    print(f"Output JSON: {OUTPUT_PATH}")

    # Load PSC items to construct CCR groups
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

    ccr_groups = construct_ccr_groups(raw_items)
    print(f"CCR groups (>=4 distinct-state clips): {len(ccr_groups)}")
    if len(ccr_groups) < 3:
        print("ERROR: Not enough CCR groups for leave-one-out evaluation.")
        return

    all_conds = get_conditions()
    if args.only_conditions:
        all_conds = [c for c in all_conds if c[0] in args.only_conditions]
    print(f"\nConditions: {[c[0] for c in all_conds]}")

    all_results = {}
    if OUTPUT_PATH.exists() and not args.force_recompute:
        with open(OUTPUT_PATH) as f:
            all_results = json.load(f)
        print(f"Resuming: existing results for {list(all_results.keys())}")

    # Sweep
    for cond_name, ckpt_path, ckpt_key in all_conds:
        if cond_name in all_results and not args.force_recompute:
            print(f"\n[SKIP] {cond_name} already in results.")
            continue

        print(f"\n{'=' * 80}")
        print(f"CONDITION: {cond_name}")
        print(f"{'=' * 80}")

        cache_path = FEATURE_CACHE_DIR / f"{cond_name}.pt"
        if not cache_path.exists():
            print(f"  [{cond_name}] ERROR: feature cache not found at {cache_path}")
            print(f"  Run sweep_ccr_only_per_epoch.py first to populate the cache.")
            continue

        print(f"  [{cond_name}] Loading cached features from {cache_path.name}")
        features_cache = torch.load(cache_path, map_location='cpu')
        print(f"  [{cond_name}] Cached features: {len(features_cache)}")

        result = run_ccr_for_condition(
            cond_name, features_cache, ccr_groups,
            n_seeds=args.n_seeds, margin=args.margin)

        all_results[cond_name] = result

        with open(OUTPUT_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [{cond_name}] Saved to {OUTPUT_PATH}")

    print_summary(all_results)
    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()