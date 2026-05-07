"""
sweep_vsd_ted_visual_per_epoch.py — VSD + TED-Visual Probe Sweep Across Adaptation Epochs

Wraps train_vsd_pad_probes.py and train_ted_visual_siamese_probe.py: imports their
helpers, loops over base + 7 saved epoch checkpoints, runs both probes per condition.

Cache strategy:
  - VSD shares feature space with PSC. Re-uses /home/tacit_experiment/_features_cache/{cond}.pt
    if present (built by the PSC sweep). Falls back to fresh extraction otherwise.
  - TED-Visual uses different clips (triplets from a separate CSV), so it builds its own
    per-condition cache at /home/tacit_experiment/_features_cache_ted_visual/{cond}.pt.

Outputs:
  /home/tacit_experiment/vsd_per_epoch_results.json
  /home/tacit_experiment/ted_visual_per_epoch_results.json

Usage (from /home/tacit_experiment):
  python -u sweep_vsd_ted_visual_per_epoch.py \
    --psc-csv /home/tacit_annotation/tacit_training_op_2026-05-02_with_clipid.csv \
    --triplets-csv /home/tacit_annotation/ted_visual.csv \
    --video-root /home/downloads

Add --skip-vsd or --skip-ted-visual to run one only.
Add --only-conditions base epoch_3 epoch_6 to limit conditions.
Add --force-recompute to ignore caches.
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
# Imports from existing probe scripts. Both modules live in /home/tacit_experiment.
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, '/home/tacit_experiment')

# VSD / PAD probe — shared encoder helpers + run_vsd
from train_vsd_pad_probes import (
    DEVICE, WINDOW_SEC, N_FOLDS,
    VSD_PAIRS, VALID_OP_LABELS,
    extract_frames_at_timestamp, build_encoder, load_checkpoint,
    encode_video, find_video_path,
    run_vsd, load_psc_data,
)

# TED-Visual siamese probe
import train_ted_visual_siamese_probe as ted_mod
from train_ted_visual_siamese_probe import (
    train_eval_siamese_groupkfold,
    summarize,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("/home/tacit_experiment/adapted_v3_ema_motion_checkpoints")
BASE_CHECKPOINT = "/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"

PSC_CACHE_DIR = Path("/home/tacit_experiment/_features_cache")
TED_CACHE_DIR = Path("/home/tacit_experiment/_features_cache_ted_visual")
TED_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_conditions():
    """(name, ckpt_path, ckpt_key) for base + all 7 epochs."""
    conds = [("base", BASE_CHECKPOINT, "ema_encoder")]
    for e in range(1, 8):
        ckpt = CHECKPOINT_DIR / f"adapted_v3_ema_motion_epoch{e:02d}.pth"
        conds.append((f"epoch_{e}", str(ckpt), "model_state"))
    return conds


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction with caching (shared across both probes per condition)
# ─────────────────────────────────────────────────────────────────────────────

def load_psc_features(cond_name):
    """Return PSC feature cache for a condition if it exists from the PSC sweep, else None."""
    cache_path = PSC_CACHE_DIR / f"{cond_name}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location='cpu')
    return None


def get_or_extract_features(cond_name, ckpt_path, ckpt_key, items, cache_path,
                             key_fn, force=False):
    """
    Generic per-condition feature extraction with disk caching.

    items: list of dicts; each dict must have a video_path or anchor_path etc.
    key_fn: function(item) -> hashable key for cache lookup
            and function(item) -> (path, timestamp) tuple for extraction
    """
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

    print(f"  [{cond_name}] Extracting features for {len(items)} items...")
    t0 = time.time()
    cache = {}
    for i, item in enumerate(items):
        key, path, ts = key_fn(item)
        if key not in cache:
            frames = extract_frames_at_timestamp(path, ts, window_sec=WINDOW_SEC)
            cache[key] = encode_video(encoder, frames)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(items)} ({time.time()-t0:.0f}s)")
    print(f"  [{cond_name}] Cached {len(cache)} unique features in {time.time()-t0:.0f}s")

    del encoder
    torch.cuda.empty_cache()

    torch.save(cache, cache_path)
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# VSD per condition
# ─────────────────────────────────────────────────────────────────────────────

def run_vsd_for_condition(cond_name, ckpt_path, ckpt_key, psc_items,
                           video_to_group, force=False):
    """Run VSD for one condition. Re-uses PSC cache if present, else extracts."""

    # Try the PSC sweep's cache first (covers all PSC items, including all VSD pairs)
    psc_cache = load_psc_features(cond_name)

    if psc_cache is None:
        # Build our own cache. VSD only needs items whose label appears in VSD_PAIRS.
        vsd_label_set = set()
        for label_a, label_b, _, _ in VSD_PAIRS:
            vsd_label_set.add(label_a)
            vsd_label_set.add(label_b)
        vsd_items = [it for it in psc_items if it['label'] in vsd_label_set]

        cache_path = PSC_CACHE_DIR / f"{cond_name}_vsdonly.pt"

        def key_fn(item):
            key = (item['video_file'], item['timestamp'])
            return key, item['video_path'], item['timestamp']

        psc_cache = get_or_extract_features(
            cond_name, ckpt_path, ckpt_key, vsd_items, cache_path, key_fn, force=force)

        if psc_cache is None:
            return None

    # run_vsd expects a features_cache keyed by (video_file, timestamp), which is
    # exactly what we have. Direct reuse.
    print(f"\n{'-' * 60}")
    print(f"  VSD — condition: {cond_name}")
    print(f"{'-' * 60}")

    return run_vsd(psc_cache, psc_items, video_to_group, cond_name)


# ─────────────────────────────────────────────────────────────────────────────
# TED-Visual per condition
# ─────────────────────────────────────────────────────────────────────────────

def load_triplets(triplets_csv, video_root):
    """Mirror the loader in train_ted_visual_siamese_probe.main()."""
    triplets_raw = []
    with open(triplets_csv) as f:
        for row in csv.DictReader(f):
            anchor_path = find_video_path(row['anchor_video'], video_root)
            correct_path = find_video_path(row['correct_video'], video_root)
            distractor_path = find_video_path(row['distractor_video'], video_root)
            if not (anchor_path and correct_path and distractor_path):
                continue

            anchor_state = row.get('anchor_state', '')
            correct_state = row.get('correct_state', '')
            distractor_state = row.get('distractor_state', '')
            strict_hard = (
                row['difficulty'] == 'hard' and
                anchor_state and correct_state and distractor_state and
                anchor_state == correct_state == distractor_state
            )

            triplets_raw.append({
                'triplet_id': row['triplet_id'],
                'difficulty': row['difficulty'],
                'anchor_video': row['anchor_video'],
                'anchor_path': anchor_path,
                'anchor_ts': float(row['anchor_ts']),
                'correct_path': correct_path,
                'correct_ts': float(row['correct_ts']),
                'distractor_path': distractor_path,
                'distractor_ts': float(row['distractor_ts']),
                'anchor_state': anchor_state,
                'correct_state': correct_state,
                'distractor_state': distractor_state,
                'strict_hard': strict_hard,
            })

    return triplets_raw


def extract_ted_visual_features(cond_name, ckpt_path, ckpt_key, triplets_raw, force=False):
    """Build a (path, ts) → feature cache for all unique triplet clips."""
    cache_path = TED_CACHE_DIR / f"{cond_name}.pt"
    if cache_path.exists() and not force:
        print(f"  [{cond_name}] Loading cached TED-Visual features from {cache_path.name}")
        return torch.load(cache_path, map_location='cpu')

    if not Path(ckpt_path).exists():
        print(f"  [{cond_name}] SKIP: checkpoint not found at {ckpt_path}")
        return None

    print(f"  [{cond_name}] Loading encoder from {Path(ckpt_path).name}...")
    encoder = build_encoder()
    encoder = load_checkpoint(encoder, ckpt_path, ckpt_key)
    encoder = encoder.to(DEVICE).eval()

    unique_clips = set()
    for t in triplets_raw:
        unique_clips.add((t['anchor_path'], t['anchor_ts']))
        unique_clips.add((t['correct_path'], t['correct_ts']))
        unique_clips.add((t['distractor_path'], t['distractor_ts']))

    print(f"  [{cond_name}] Extracting features for {len(unique_clips)} unique triplet clips...")
    cache = {}
    t0 = time.time()
    for i, (path, ts) in enumerate(sorted(unique_clips)):
        frames = extract_frames_at_timestamp(path, ts, window_sec=WINDOW_SEC)
        cache[(path, ts)] = encode_video(encoder, frames)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(unique_clips)} ({time.time()-t0:.0f}s)")

    print(f"  [{cond_name}] Extracted {len(cache)} features in {time.time()-t0:.0f}s")
    del encoder
    torch.cuda.empty_cache()

    torch.save(cache, cache_path)
    return cache


def run_ted_visual_for_condition(cond_name, ckpt_path, ckpt_key, triplets_raw,
                                   n_seeds=3, margin=0.2, force=False):
    """Run TED-Visual siamese probe for one condition with multi-seed averaging."""

    features_cache = extract_ted_visual_features(
        cond_name, ckpt_path, ckpt_key, triplets_raw, force=force)
    if features_cache is None:
        return None

    triplets_data = []
    for t in triplets_raw:
        triplets_data.append({
            **t,
            'anchor_feat': features_cache[(t['anchor_path'], t['anchor_ts'])],
            'correct_feat': features_cache[(t['correct_path'], t['correct_ts'])],
            'distractor_feat': features_cache[(t['distractor_path'], t['distractor_ts'])],
        })

    print(f"\n{'-' * 60}")
    print(f"  TED-Visual — condition: {cond_name}  ({n_seeds} seeds)")
    print(f"{'-' * 60}")

    seed_results = []
    for seed in range(n_seeds):
        torch.manual_seed(42 + seed)
        np.random.seed(42 + seed)
        print(f"\n  Seed {seed + 1}/{n_seeds}")
        # Reset the global video→group map between seeds so results are reproducible.
        # (The probe script uses a module-level dict; clearing it ensures no leakage.)
        ted_mod._video_to_group = {}
        res = train_eval_siamese_groupkfold(
            triplets_data, n_folds=5, margin=margin,
            condition_name=f"{cond_name}-s{seed}",
        )
        if res is not None:
            seed_results.append(res)

    if not seed_results:
        print(f"  [{cond_name}] WARNING: no successful folds")
        return None

    cond_results = {}
    for probe_name in ['siamese', 'cos_projected', 'cos_raw', 'anchor_ablation']:
        per_seed_acc = [r[probe_name].mean() for r in seed_results]
        mean_arr = np.mean([r[probe_name] for r in seed_results], axis=0) > 0.5
        cond_results[probe_name] = summarize(mean_arr, triplets_data)
        cond_results[probe_name]['overall_mean'] = round(float(np.mean(per_seed_acc)), 3)
        cond_results[probe_name]['overall_std'] = round(float(np.std(per_seed_acc)), 3)

    s = cond_results['siamese']
    a = cond_results['anchor_ablation']
    print(f"\n  [{cond_name}] Siamese:        {s['overall_mean']:.3f} ± {s['overall_std']:.3f}")
    print(f"  [{cond_name}] Anchor ablation: {a['overall_mean']:.3f} ± {a['overall_std']:.3f}")
    if 'hard_strict' in s:
        print(f"  [{cond_name}] Strict Hard:     {s['hard_strict']['accuracy']:.3f} "
              f"(n={s['hard_strict']['n']})")

    return cond_results


# ─────────────────────────────────────────────────────────────────────────────
# Summary table printers
# ─────────────────────────────────────────────────────────────────────────────

def print_vsd_summary(vsd_results):
    print(f"\n\n{'=' * 90}")
    print("VSD SWEEP SUMMARY — Per-Pair and Mean Accuracy Across Adaptation Epochs")
    print(f"{'=' * 90}")

    if not vsd_results:
        print("(no VSD results)")
        return

    all_pairs = set()
    for r in vsd_results.values():
        if r:
            all_pairs.update(r.get('per_pair', {}).keys())

    print(f"\n{'Pair':<55}", end="")
    for cond in vsd_results:
        print(f" {cond:>10}", end="")
    print()
    print("-" * (55 + 11 * len(vsd_results)))

    for pair in sorted(all_pairs):
        print(f"{pair[:55]:<55}", end="")
        for cond, r in vsd_results.items():
            if r is None:
                print(f" {'-':>10}", end="")
            else:
                acc = r.get('per_pair', {}).get(pair, {}).get('overall_acc', None)
                print(f" {acc:>10.3f}" if acc is not None else f" {'-':>10}", end="")
        print()

    print("-" * (55 + 11 * len(vsd_results)))
    print(f"{'MEAN':<55}", end="")
    for cond, r in vsd_results.items():
        if r is None:
            print(f" {'-':>10}", end="")
        else:
            print(f" {r.get('mean_acc', 0):>10.3f}", end="")
    print()

    # Best epoch by mean
    scored = {c: (r['mean_acc'] if r else -1) for c, r in vsd_results.items()}
    best = max(scored, key=scored.get)
    print(f"\nBest epoch by mean VSD accuracy: {best} ({scored[best]:.3f})")


def print_ted_visual_summary(ted_results):
    print(f"\n\n{'=' * 90}")
    print("TED-VISUAL SWEEP SUMMARY — Siamese Probe Accuracy Across Adaptation Epochs")
    print(f"{'=' * 90}")

    if not ted_results:
        print("(no TED-Visual results)")
        return

    print(f"\n{'Condition':<12} {'Easy':>10} {'Medium':>10} {'Hard':>10} "
          f"{'Strict-H':>10} {'Soft-H':>10} {'Overall':>14}  {'AnchAbl':>10}")
    print("-" * 95)

    for cond, r in ted_results.items():
        if r is None:
            print(f"{cond:<12}  (no result)")
            continue
        s = r['siamese']
        a = r['anchor_ablation']
        easy = s.get('easy', {}).get('accuracy', None)
        med = s.get('medium', {}).get('accuracy', None)
        hard = s.get('hard', {}).get('accuracy', None)
        strict = s.get('hard_strict', {}).get('accuracy', None)
        soft = s.get('hard_soft', {}).get('accuracy', None)
        overall_m = s.get('overall_mean')
        overall_s = s.get('overall_std')
        anch = a.get('overall_mean')

        def fmt(v):
            return f"{v:>10.3f}" if v is not None else f"{'-':>10}"

        overall_str = f"{overall_m:.3f}±{overall_s:.3f}" if overall_m is not None else "-"
        print(f"{cond:<12} {fmt(easy)} {fmt(med)} {fmt(hard)} {fmt(strict)} {fmt(soft)} "
              f"  {overall_str:>14}  {fmt(anch)}")

    # Best per subset
    print("\nBest epoch per subset (siamese accuracy):")
    for subset_key, label in [
        ('overall_mean', 'Overall'),
        (('hard_strict', 'accuracy'), 'Strict Hard (load-bearing)'),
        (('easy', 'accuracy'), 'Easy'),
        (('medium', 'accuracy'), 'Medium'),
        (('hard', 'accuracy'), 'Hard (aggregate)'),
    ]:
        scores = {}
        for c, r in ted_results.items():
            if r is None:
                continue
            s = r['siamese']
            if isinstance(subset_key, tuple):
                v = s.get(subset_key[0], {}).get(subset_key[1])
            else:
                v = s.get(subset_key)
            if v is not None:
                scores[c] = v
        if scores:
            best = max(scores, key=scores.get)
            print(f"  {label:<28} best at {best:<10} ({scores[best]:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psc-csv", required=True,
                        help="PSC annotations CSV (used for VSD)")
    parser.add_argument("--triplets-csv", required=True,
                        help="TED-Visual triplets CSV")
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--skip-vsd", action="store_true")
    parser.add_argument("--skip-ted-visual", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Seeds for TED-Visual multi-seed averaging")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="TED-Visual triplet margin")
    parser.add_argument("--only-conditions", nargs="+", default=None)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    print("=" * 90)
    print("TACIT — VSD + TED-Visual Probe Sweep Across Epochs")
    print("=" * 90)
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_SEC}s")
    print(f"Run VSD: {not args.skip_vsd}")
    print(f"Run TED-Visual: {not args.skip_ted_visual}")
    print(f"PSC cache dir: {PSC_CACHE_DIR}")
    print(f"TED cache dir: {TED_CACHE_DIR}")

    # Load PSC items (for VSD) using the existing loader
    psc_items = []
    if not args.skip_vsd:
        psc_items = load_psc_data(args.psc_csv, args.video_root)
        print(f"\nVSD: loaded {len(psc_items)} PSC items "
              f"from {len(set(it['video_file'] for it in psc_items))} videos")

    # Load triplets (for TED-Visual)
    triplets_raw = []
    if not args.skip_ted_visual:
        if not Path(args.triplets_csv).exists():
            print(f"\nWARNING: triplets CSV not found: {args.triplets_csv}")
            print("Skipping TED-Visual.")
            args.skip_ted_visual = True
        else:
            triplets_raw = load_triplets(args.triplets_csv, args.video_root)
            print(f"\nTED-Visual: loaded {len(triplets_raw)} triplets")
            diff_counts = defaultdict(int)
            strict_count = 0
            for t in triplets_raw:
                diff_counts[t['difficulty']] += 1
                if t['strict_hard']:
                    strict_count += 1
            for d in ['easy', 'medium', 'hard']:
                print(f"  {d}: {diff_counts[d]}")
            print(f"  hard_strict: {strict_count} (subset of hard)")
            print(f"  hard_soft:   {diff_counts['hard'] - strict_count}")

    # Conditions
    all_conds = get_conditions()
    if args.only_conditions:
        all_conds = [c for c in all_conds if c[0] in args.only_conditions]
    print(f"\nConditions: {[c[0] for c in all_conds]}")

    # video_to_group for VSD
    video_to_group = {}
    if psc_items:
        all_videos = sorted(set(it['video_file'] for it in psc_items))
        video_to_group = {v: i for i, v in enumerate(all_videos)}

    # Resume from existing partial results if present
    vsd_path = Path("/home/tacit_experiment/vsd_per_epoch_results.json")
    ted_path = Path("/home/tacit_experiment/ted_visual_per_epoch_results.json")

    vsd_results = {}
    if vsd_path.exists():
        with open(vsd_path) as f:
            vsd_results = json.load(f)
        print(f"\nResuming VSD: existing conditions {list(vsd_results.keys())}")

    ted_results = {}
    if ted_path.exists():
        with open(ted_path) as f:
            ted_results = json.load(f)
        print(f"Resuming TED-Visual: existing conditions {list(ted_results.keys())}")

    # Sweep
    for cond_name, ckpt_path, ckpt_key in all_conds:
        print(f"\n{'=' * 90}")
        print(f"CONDITION: {cond_name}")
        print(f"{'=' * 90}")

        # VSD
        if not args.skip_vsd:
            if cond_name in vsd_results and not args.force_recompute:
                print(f"  [SKIP] {cond_name} already has VSD results")
            else:
                vsd_r = run_vsd_for_condition(
                    cond_name, ckpt_path, ckpt_key,
                    psc_items, video_to_group, force=args.force_recompute)
                vsd_results[cond_name] = vsd_r
                with open(vsd_path, 'w') as f:
                    json.dump(vsd_results, f, indent=2)
                print(f"  [{cond_name}] VSD saved")

        # TED-Visual
        if not args.skip_ted_visual:
            if cond_name in ted_results and not args.force_recompute:
                print(f"  [SKIP] {cond_name} already has TED-Visual results")
            else:
                ted_r = run_ted_visual_for_condition(
                    cond_name, ckpt_path, ckpt_key, triplets_raw,
                    n_seeds=args.n_seeds, margin=args.margin,
                    force=args.force_recompute)
                ted_results[cond_name] = ted_r
                with open(ted_path, 'w') as f:
                    json.dump(ted_results, f, indent=2)
                print(f"  [{cond_name}] TED-Visual saved")

    # Final summaries
    if not args.skip_vsd:
        print_vsd_summary(vsd_results)
    if not args.skip_ted_visual:
        print_ted_visual_summary(ted_results)

    print(f"\n{'=' * 90}")
    print("DONE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()