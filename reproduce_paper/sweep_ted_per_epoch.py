"""
sweep_ted_per_epoch.py — TED Probe Sweep Across Adaptation Epochs

Wraps train_ted_probes.py: imports its helpers, loops over base + 7 saved
epoch checkpoints, runs both Variant A (visual-only) and Variant B (visual+text)
per condition.

Cache strategy:
  - PSC features: tries to reuse _features_cache/{cond}.pt from the PSC sweep
    if present. Falls back to fresh extraction.
  - TED features: separate per-condition cache at _features_cache_ted/{cond}.pt
    since TED items differ from PSC items.
  - CLIP text features: encoded once at the top, reused across all conditions
    (encoder-independent).

Output: ted_per_epoch_results.json

Usage (from /home/tacit_experiment):
  python -u sweep_ted_per_epoch.py \
    --psc-csv /home/tacit_annotation/tacit_training_op_2026-05-02_with_clipid.csv \
    --ted-csv /home/tacit_annotation/ted_op.csv \
    --video-root /home/downloads \
    2>&1 | grep --line-buffered -v "av1\\|Failed to get pixel\\|hardware accelerated\\|Get current frame" \
    | tee ted_per_epoch_results.log

Add --skip-visual-text to run Variant A only (faster; no CLIP needed).
Add --only-conditions base epoch_4 epoch_6 to limit conditions.
Add --force-recompute to ignore caches.
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
# Imports from existing TED probe script — no re-implementation
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, '/home/tacit_experiment')

from train_ted_probes import (
    DEVICE, WINDOW_SEC, N_FOLDS,
    VALID_OP_LABELS, LABEL_TO_IDX,
    extract_frames_at_timestamp,
    build_encoder, load_checkpoint, encode_video,
    build_text_encoder, encode_text,
    train_visual_only_psc, evaluate_visual_only_ted,
    train_and_eval_visual_text_groupkfold,
    load_psc_data, load_ted_data,
    find_video_path,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("/home/tacit_experiment/adapted_v3_ema_motion_checkpoints")
BASE_CHECKPOINT = "/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"

PSC_CACHE_DIR = Path("/home/tacit_experiment/_features_cache")        # from PSC sweep
TED_CACHE_DIR = Path("/home/tacit_experiment/_features_cache_ted")    # new, separate
TED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = Path("/home/tacit_experiment/ted_per_epoch_results.json")


def get_conditions():
    """(name, ckpt_path, ckpt_key) for base + all 7 epochs."""
    conds = [("base", BASE_CHECKPOINT, "ema_encoder")]
    for e in range(1, 8):
        ckpt = CHECKPOINT_DIR / f"adapted_v3_ema_motion_epoch{e:02d}.pth"
        conds.append((f"epoch_{e}", str(ckpt), "model_state"))
    return conds


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction with separate caches for PSC and TED items
# ─────────────────────────────────────────────────────────────────────────────

def get_psc_features(cond_name, ckpt_path, ckpt_key, psc_items, force=False):
    """Get PSC features for a condition. Reuses PSC sweep cache if present."""
    cache_path = PSC_CACHE_DIR / f"{cond_name}.pt"
    if cache_path.exists() and not force:
        print(f"  [{cond_name}] PSC: loading cached features from {cache_path.name}")
        cache = torch.load(cache_path, map_location='cpu')

        # Verify all needed keys are present
        missing = [it for it in psc_items
                   if (it['video_file'], it['timestamp']) not in cache]
        if missing:
            print(f"  [{cond_name}] PSC: WARNING {len(missing)} items missing from cache; falling back to fresh extraction")
        else:
            return cache, None  # None = encoder not loaded (we'll need it later for TED)

    # Fall through: extract fresh (encoder will be loaded and reused for TED)
    return None, None


def extract_features_for_items(encoder, items, label):
    """Extract features for a list of items. Returns dict keyed by (video, ts)."""
    cache = {}
    t0 = time.time()
    for i, item in enumerate(items):
        key = (item['video_file'], item['timestamp'])
        if key in cache:
            continue
        frames = extract_frames_at_timestamp(item['video_path'], item['timestamp'])
        cache[key] = encode_video(encoder, frames)
        if (i + 1) % 50 == 0:
            print(f"    {label} {i+1}/{len(items)} ({time.time()-t0:.0f}s)")
    print(f"  Extracted {len(cache)} {label} features in {time.time()-t0:.0f}s")
    return cache


def get_or_extract_features(cond_name, ckpt_path, ckpt_key, psc_items, ted_items,
                              force=False):
    """
    Returns (psc_features_cache, ted_features_cache).
    Reuses on-disk caches where present; loads encoder once if either needs extraction.
    """
    psc_cache_path = PSC_CACHE_DIR / f"{cond_name}.pt"
    ted_cache_path = TED_CACHE_DIR / f"{cond_name}.pt"

    psc_cache = None
    ted_cache = None
    psc_needs_extraction = False
    ted_needs_extraction = False

    # Check PSC cache
    if psc_cache_path.exists() and not force:
        print(f"  [{cond_name}] PSC: loading {psc_cache_path.name}")
        cand = torch.load(psc_cache_path, map_location='cpu')
        # Verify completeness
        missing = [it for it in psc_items
                   if (it['video_file'], it['timestamp']) not in cand]
        if not missing:
            psc_cache = cand
        else:
            print(f"  [{cond_name}] PSC: cache incomplete ({len(missing)} missing), re-extracting")
            psc_needs_extraction = True
    else:
        psc_needs_extraction = True

    # Check TED cache
    if ted_cache_path.exists() and not force:
        print(f"  [{cond_name}] TED: loading {ted_cache_path.name}")
        cand = torch.load(ted_cache_path, map_location='cpu')
        missing = [it for it in ted_items
                   if (it['video_file'], it['timestamp']) not in cand]
        if not missing:
            ted_cache = cand
        else:
            print(f"  [{cond_name}] TED: cache incomplete ({len(missing)} missing), re-extracting")
            ted_needs_extraction = True
    else:
        ted_needs_extraction = True

    # If anything needs extraction, load encoder once
    if psc_needs_extraction or ted_needs_extraction:
        if not Path(ckpt_path).exists():
            print(f"  [{cond_name}] SKIP: checkpoint not found at {ckpt_path}")
            return None, None

        print(f"  [{cond_name}] Loading encoder from {Path(ckpt_path).name}...")
        encoder = build_encoder()
        encoder = load_checkpoint(encoder, ckpt_path, ckpt_key)
        encoder = encoder.to(DEVICE).eval()

        if psc_needs_extraction:
            print(f"  [{cond_name}] Extracting PSC features ({len(psc_items)})...")
            psc_cache = extract_features_for_items(encoder, psc_items, "PSC")
            torch.save(psc_cache, psc_cache_path)

        if ted_needs_extraction:
            print(f"  [{cond_name}] Extracting TED features ({len(ted_items)})...")
            ted_cache = extract_features_for_items(encoder, ted_items, "TED")
            torch.save(ted_cache, ted_cache_path)

        del encoder
        torch.cuda.empty_cache()

    return psc_cache, ted_cache


# ─────────────────────────────────────────────────────────────────────────────
# Run TED probes for one condition
# ─────────────────────────────────────────────────────────────────────────────

def run_ted_probes_for_condition(cond_name, psc_cache, ted_cache,
                                   psc_items, ted_items,
                                   question_features, option_features,
                                   skip_visual_text=False):
    """Run Variant A + Variant B for one V-JEPA condition."""
    # Build aligned tensors from caches
    psc_features = torch.stack([psc_cache[(it['video_file'], it['timestamp'])]
                                for it in psc_items])
    psc_labels = torch.tensor([LABEL_TO_IDX[it['label']] for it in psc_items],
                              dtype=torch.long)

    ted_features = torch.stack([ted_cache[(it['video_file'], it['timestamp'])]
                                for it in ted_items])
    ted_correct = torch.tensor([it['correct_idx'] for it in ted_items], dtype=torch.long)
    ted_options_indices = [it['option_indices'] for it in ted_items]

    # Groups for V+T probe (TED-side groups)
    all_videos = sorted(set(it['video_file'] for it in psc_items + ted_items))
    video_to_group = {v: i for i, v in enumerate(all_videos)}
    ted_groups = np.array([video_to_group[it['video_file']] for it in ted_items])

    cond_results = {}

    # ── Variant A: Visual-only ──────────────────────────────────────────
    print(f"\n  [{cond_name}] --- Variant A: Visual-Only ---")
    print(f"  [{cond_name}] Training PSC probe on {len(psc_features)} items...")
    psc_probe = train_visual_only_psc(
        psc_features, psc_labels, n_classes=len(VALID_OP_LABELS),
        n_epochs=100, lr=1e-3)

    print(f"  [{cond_name}] Evaluating on TED...")
    visual_only_acc = evaluate_visual_only_ted(
        psc_probe, ted_features, ted_options_indices, ted_correct.tolist())
    print(f"  [{cond_name}] Visual-only TED accuracy: {visual_only_acc:.3f}")
    cond_results['visual_only'] = {'accuracy': round(visual_only_acc, 3)}

    del psc_probe
    torch.cuda.empty_cache()

    # ── Variant B: Visual + Text ────────────────────────────────────────
    if not skip_visual_text:
        print(f"\n  [{cond_name}] --- Variant B: Visual + Text ---")
        vt_result = train_and_eval_visual_text_groupkfold(
            ted_features, question_features, option_features,
            ted_correct, ted_groups,
            n_folds=N_FOLDS, condition_name=cond_name)
        cond_results['visual_text'] = vt_result

    return cond_results


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results):
    print(f"\n\n{'=' * 80}")
    print("TED SWEEP SUMMARY — Across Adaptation Epochs")
    print(f"{'=' * 80}")
    print(f"\n  Random baseline: 0.250 (1 of 4 options)\n")

    if not all_results:
        print("(no results)")
        return

    print(f"{'Condition':<12} {'V-only':>10} {'V+T overall':>14}")
    print("-" * 40)

    for cond, r in all_results.items():
        v_only = r.get('visual_only', {}).get('accuracy', None)
        v_t = r.get('visual_text', {}).get('overall_acc', None)

        def fmt(v, w=10, prec=3):
            return f"{v:>{w}.{prec}f}" if v is not None else f"{'-':>{w}}"

        print(f"{cond:<12} {fmt(v_only)} {fmt(v_t, 14)}")

    print("\nBest epoch per variant:")
    for variant, key, label in [
        ('visual_only', 'accuracy', 'Visual-only'),
        ('visual_text', 'overall_acc', 'Visual+Text'),
    ]:
        scores = {c: r.get(variant, {}).get(key, -1) for c, r in all_results.items()}
        scores = {c: v for c, v in scores.items() if v != -1}
        if scores:
            best = max(scores, key=scores.get)
            print(f"  {label:<14} best at {best:<10} ({scores[best]:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psc-csv", required=True)
    parser.add_argument("--ted-csv", required=True)
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--skip-visual-text", action="store_true",
                        help="Skip Variant B (saves time; no CLIP needed)")
    parser.add_argument("--only-conditions", nargs="+", default=None,
                        help="Run only these conditions (e.g. 'base epoch_4 epoch_6')")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore caches and existing results")
    args = parser.parse_args()

    print("=" * 80)
    print("TACIT — TED Probe Sweep Across Adaptation Epochs")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_SEC}s")
    print(f"PSC cache dir: {PSC_CACHE_DIR}")
    print(f"TED cache dir: {TED_CACHE_DIR}")
    print(f"Output JSON: {OUTPUT_PATH}")

    # Load annotations
    psc_items = load_psc_data(args.psc_csv, args.video_root)
    ted_items = load_ted_data(args.ted_csv, args.video_root)
    print(f"\nPSC items: {len(psc_items)} from {len(set(it['video_file'] for it in psc_items))} videos")
    print(f"TED items: {len(ted_items)} from {len(set(it['video_file'] for it in ted_items))} videos")

    if len(ted_items) < 30:
        print("\nWARNING: Few TED items. Results will be noisy.")

    # ── Encode CLIP text features once (encoder-independent) ──────────
    question_features = None
    option_features = None
    if not args.skip_visual_text:
        try:
            print(f"\nLoading CLIP text encoder...")
            clip_model = build_text_encoder()
            print(f"  CLIP loaded.")

            print(f"Encoding TED questions and options...")
            questions_text = [it['question'] for it in ted_items]
            question_features = encode_text(clip_model, questions_text)

            all_option_features = []
            for it in ted_items:
                opt_feats = encode_text(clip_model, it['options'])
                all_option_features.append(opt_feats)
            option_features = torch.stack(all_option_features)

            print(f"  Question features: {question_features.shape}")
            print(f"  Option features:   {option_features.shape}")

            del clip_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  CLIP load/encoding failed: {e}")
            print(f"  Skipping Variant B for the rest of the sweep.")
            args.skip_visual_text = True

    # ── Conditions ────────────────────────────────────────────────────
    all_conds = get_conditions()
    if args.only_conditions:
        all_conds = [c for c in all_conds if c[0] in args.only_conditions]
    print(f"\nConditions: {[c[0] for c in all_conds]}")

    # Resume from existing results
    all_results = {}
    if OUTPUT_PATH.exists() and not args.force_recompute:
        with open(OUTPUT_PATH) as f:
            all_results = json.load(f)
        print(f"Resuming: existing TED results for {list(all_results.keys())}")

    # ── Sweep ─────────────────────────────────────────────────────────
    for cond_name, ckpt_path, ckpt_key in all_conds:
        if cond_name in all_results and not args.force_recompute:
            existing = all_results[cond_name]
            has_v_only = 'visual_only' in existing
            has_v_t = 'visual_text' in existing or args.skip_visual_text
            if has_v_only and has_v_t:
                print(f"\n[SKIP] {cond_name} already has all requested TED results.")
                continue

        print(f"\n{'=' * 80}")
        print(f"CONDITION: {cond_name}")
        print(f"{'=' * 80}")

        psc_cache, ted_cache = get_or_extract_features(
            cond_name, ckpt_path, ckpt_key, psc_items, ted_items,
            force=args.force_recompute)

        if psc_cache is None or ted_cache is None:
            print(f"  [{cond_name}] Skipping due to missing features")
            continue

        cond_results = run_ted_probes_for_condition(
            cond_name, psc_cache, ted_cache,
            psc_items, ted_items,
            question_features, option_features,
            skip_visual_text=args.skip_visual_text,
        )

        all_results[cond_name] = cond_results

        # Persist after every condition
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  [{cond_name}] Saved partial results to {OUTPUT_PATH}")

    print_summary(all_results)
    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()