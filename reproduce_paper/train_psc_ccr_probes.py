"""
train_psc_ccr_probes.py — Full PSC + CCR Probe Training Pipeline

Takes the merged OP annotation CSV (241 items, 38 videos) and runs:
  1. PSC probes at 3 granularity levels (23, 20, 10 classes)
  2. CCR pairwise ordering probe (288 pairs from 20 videos)
  3. Both base and adapted V-JEPA conditions
  4. 4s extraction window (best from directional tests)
  5. GroupKFold (no data leakage across videos)

Usage:
  cd /home/tacit_experiment
  python -u train_psc_ccr_probes.py \
    --psc-csv /home/tacit_annotation/tacit_training_op_2026-05-02_with_clipid.csv \
    --video-root /home/downloads \
    2>&1 | grep --line-buffered -v "av1|Failed to get pixel|hardware accelerated|Get current frame" \
    | tee psc_ccr_full_results.log
"""

import argparse
import csv
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import GroupKFold
from scipy.stats import kendalltau

sys.path.insert(0, '/home/.cache/torch/hub/facebookresearch_vjepa2_main')
sys.path.insert(0, '/home/tacit_experiment')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SEC = 4.0  # 4s window — best from directional test
N_FOLDS = 5       # 38 videos → ~7-8 per fold, workable

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# CLASS MERGING STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

MERGE_A = {
    # 23 classes — drop only titration_running (1) and tlc_plate_dry (2)
    'titration_running': None,
    'tlc_plate_dry': None,
}  # everything else maps to itself

MERGE_B = {
    # 20 classes — group distillation sub-steps and filtration types
    'distillation_setup_running': 'distillation_running',
    'reflux_running': 'distillation_running',
    'distillate_collecting': 'distillation_running',
    'vacuum_filtration_general': 'filtration',
    'gravity_filtration_hot': 'filtration',
    'titration_running': None,
    'tlc_plate_dry': None,
}

MERGE_C = {
    # 10 classes — group by process type
    'mixture_dissolved_hot': 'mixture_heating',
    'mixture_crude_unreacted': 'mixture_heating',
    'lle_draining_lower_layer': 'lle_separation',
    'lle_two_phase_settled': 'lle_separation',
    'crystals_forming': 'crystallization',
    'crystals_complete': 'crystallization',
    'distillation_setup_running': 'distillation',
    'reflux_running': 'distillation',
    'distillate_collecting': 'distillation',
    'rotovap_running': 'rotovap_evaporation',
    'fractions_collecting': 'column_chromatography',
    'sample_loaded': 'column_chromatography',
    'fraction_analysis': 'column_chromatography',
    'column_packed': 'column_chromatography',
    'column_equilibrated': 'column_chromatography',
    'column_dry': 'column_chromatography',
    'tlc_developed': 'tlc',
    'tlc_plate_spotted': 'tlc',
    'tlc_running': 'tlc',
    'tlc_plate_dry': 'tlc',
    'vacuum_filtration_general': 'filtration',
    'gravity_filtration_hot': 'filtration',
    'solvent_dispensing': 'solvent_dispensing',
    'analytical_weighing': 'analytical_weighing',
    'titration_running': None,
}


def apply_merge(raw_label, merge_map):
    """Apply a merge map. Returns None if label should be dropped."""
    if raw_label in merge_map:
        return merge_map[raw_label]
    return raw_label  # keep as-is if not in map


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

import cv2

def extract_frames_at_timestamp(video_path, timestamp_sec,
                                 n_frames=16, frame_size=384, window_sec=4.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return torch.zeros(3, n_frames, frame_size, frame_size)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps

    start_sec = max(0, timestamp_sec - window_sec / 2)
    end_sec = min(total_sec, timestamp_sec + window_sec / 2)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    if end_frame <= start_frame:
        end_frame = min(start_frame + n_frames, total_frames)

    indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size, frame_size))
        frames.append(frame)
    cap.release()

    arr = np.stack(frames).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = arr.transpose(3, 0, 1, 2)
    return torch.from_numpy(arr).float()


# ══════════════════════════════════════════════════════════════════════════════
# V-JEPA ENCODER
# ══════════════════════════════════════════════════════════════════════════════

def build_encoder():
    from app.vjepa_2_1.models import vision_transformer as vit_encoder
    return vit_encoder.__dict__['vit_large'](
        patch_size=16, img_size=(384, 384), num_frames=64,
        tubelet_size=2, use_sdpa=True, use_SiLU=False,
        wide_SiLU=True, uniform_power=False, use_rope=True,
        img_temporal_dim_size=1, interpolate_rope=True,
    )


def load_checkpoint(encoder, checkpoint_path, key=None):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if key and isinstance(ckpt, dict) and key in ckpt:
        state = ckpt[key]
    elif isinstance(ckpt, dict) and 'model_state' in ckpt:
        state = ckpt['model_state']
    elif isinstance(ckpt, dict) and 'ema_encoder' in ckpt:
        state = ckpt['ema_encoder']
        state = {k.replace('module.', '').replace('backbone.', ''): v
                 for k, v in state.items()}
    else:
        state = ckpt
    encoder.load_state_dict(state, strict=False)
    return encoder


def encode_video(encoder, frames_tensor):
    x = frames_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = encoder(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        feat = out.mean(dim=1).squeeze(0).cpu().float()
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# PSC PROBE
# ══════════════════════════════════════════════════════════════════════════════

class PSCProbe(nn.Module):
    def __init__(self, embed_dim=1024, n_classes=10):
        super().__init__()
        self.probe = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.probe(x)


def run_psc_groupkfold(features, labels, groups, class_names,
                        condition_name, n_folds=5, n_epochs=100, lr=1e-3):
    n_classes = len(class_names)
    embed_dim = features.shape[1]

    # Adjust folds if fewer groups than folds
    unique_groups = len(set(groups.tolist()))
    actual_folds = min(n_folds, unique_groups)
    if actual_folds < n_folds:
        print(f"    Reducing folds from {n_folds} to {actual_folds} (only {unique_groups} groups)")

    gkf = GroupKFold(n_splits=actual_folds)
    all_preds = torch.zeros(len(labels), dtype=torch.long)
    all_true = labels.clone()
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(features, labels, groups)):
        train_f = features[train_idx].to(DEVICE)
        train_l = labels[train_idx].to(DEVICE)
        val_f = features[val_idx].to(DEVICE)
        val_l = labels[val_idx].to(DEVICE)

        probe = PSCProbe(embed_dim, n_classes).to(DEVICE)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        best_preds = None

        for epoch in range(1, n_epochs + 1):
            probe.train()
            loss = criterion(probe(train_f), train_l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0 or epoch == n_epochs:
                probe.eval()
                with torch.no_grad():
                    preds = probe(val_f).argmax(dim=1)
                    acc = (preds == val_l).float().mean().item()
                if acc >= best_acc:
                    best_acc = acc
                    best_preds = preds.cpu()

        all_preds[val_idx] = best_preds
        fold_accs.append(best_acc)

    overall_acc = (all_preds == all_true).float().mean().item()

    per_class = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = all_true == cls_idx
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == all_true[mask]).float().mean().item()
            per_class[cls_name] = {'accuracy': round(cls_acc, 3), 'count': int(mask.sum())}

    return {
        'overall_acc': round(overall_acc, 3),
        'fold_accs': [round(a, 3) for a in fold_accs],
        'per_class': per_class,
        'n_classes': n_classes,
        'n_items': len(labels),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CCR PROBE — Pairwise Ordering
# ══════════════════════════════════════════════════════════════════════════════

class PairwiseOrderProbe(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, feat_a, feat_b):
        combined = torch.cat([feat_a, feat_b], dim=-1)
        return self.net(combined).squeeze(-1)


def construct_ccr_groups(items):
    by_video = defaultdict(list)
    for item in items:
        by_video[item['video_file']].append(item)

    groups = []
    for video_file, video_items in by_video.items():
        video_items.sort(key=lambda x: x['timestamp'])
        deduped = [video_items[0]]
        for item in video_items[1:]:
            if item['raw_label'] != deduped[-1]['raw_label']:
                deduped.append(item)
        if len(deduped) >= 4:
            groups.append({
                'video_file': video_file,
                'items': deduped,
                'n_clips': len(deduped),
            })
    return groups


def generate_pairs(clip_features):
    n = len(clip_features)
    feat_a, feat_b, labels = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                feat_a.append(clip_features[i])
                feat_b.append(clip_features[j])
                labels.append(1.0)
            else:
                feat_a.append(clip_features[j])
                feat_b.append(clip_features[i])
                labels.append(0.0)
    return torch.stack(feat_a), torch.stack(feat_b), torch.tensor(labels, dtype=torch.float)


def predict_ordering(probe, clip_features):
    n = len(clip_features)
    scores = torch.zeros(n)
    probe.eval()
    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                fa = clip_features[i].unsqueeze(0).to(DEVICE)
                fb = clip_features[j].unsqueeze(0).to(DEVICE)
                logit = probe(fa, fb)
                scores[i] += torch.sigmoid(logit).item()
    return torch.argsort(scores, descending=True).tolist()


def run_ccr_leave_one_out(group_features, groups, condition_name):
    all_taus = []
    all_pair_accs = []

    for held_out in range(len(groups)):
        train_fa, train_fb, train_lab = [], [], []
        for gi in range(len(groups)):
            if gi == held_out:
                continue
            fa, fb, lab = generate_pairs(group_features[gi])
            train_fa.append(fa)
            train_fb.append(fb)
            train_lab.append(lab)

        train_fa = torch.cat(train_fa).to(DEVICE)
        train_fb = torch.cat(train_fb).to(DEVICE)
        train_lab = torch.cat(train_lab).to(DEVICE)

        probe = PairwiseOrderProbe(1024).to(DEVICE)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(100):
            probe.train()
            logits = probe(train_fa, train_fb)
            loss = criterion(logits, train_lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on held-out
        held_feats = group_features[held_out]
        n_clips = len(held_feats)
        true_order = list(range(n_clips))

        probe.eval()
        pair_correct = 0
        pair_total = 0
        with torch.no_grad():
            for i in range(n_clips):
                for j in range(i + 1, n_clips):
                    fa = held_feats[i].unsqueeze(0).to(DEVICE)
                    fb = held_feats[j].unsqueeze(0).to(DEVICE)
                    logit = probe(fa, fb)
                    if (torch.sigmoid(logit) > 0.5).item():
                        pair_correct += 1
                    pair_total += 1

        pair_acc = pair_correct / pair_total if pair_total > 0 else 0
        predicted_order = predict_ordering(probe, held_feats)
        tau, _ = kendalltau(true_order, predicted_order)
        if np.isnan(tau):
            tau = 0.0

        all_taus.append(tau)
        all_pair_accs.append(pair_acc)

    mean_tau = np.mean(all_taus)
    std_tau = np.std(all_taus)
    mean_pair = np.mean(all_pair_accs)

    print(f"    [{condition_name}] CCR: tau={mean_tau:.3f}±{std_tau:.3f}, pair_acc={mean_pair:.3f}")

    return {
        'mean_tau': round(float(mean_tau), 3),
        'std_tau': round(float(std_tau), 3),
        'mean_pairwise_acc': round(float(mean_pair), 3),
        'per_group_tau': [round(float(t), 3) for t in all_taus],
        'n_groups': len(groups),
        'n_total_pairs': sum(g['n_clips'] * (g['n_clips'] - 1) // 2 for g in groups),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIND VIDEO PATH
# ══════════════════════════════════════════════════════════════════════════════

def find_video_path(video_file, video_root):
    candidates = [
        Path(video_root) / "organic_purification" / "youtube" / video_file,
        Path(video_root) / "organic_purification" / "youtube" / video_file.replace('good_', ''),
        Path(video_root) / "organic_purification" / video_file,
        Path(video_root) / video_file,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    clean_name = video_file.replace('good_', '').replace('ok_', '').replace('bad_', '')
    found = list(Path(video_root).rglob(clean_name))
    if found:
        return str(found[0])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psc-csv", required=True)
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--base-checkpoint",
                        default="/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt")
    parser.add_argument("--adapted-checkpoint",
                        default="/home/tacit_experiment/adapted_v3_ema_motion_checkpoints/adapted_v3_ema_motion_epoch05.pth")
    parser.add_argument("--adapted-key", default="model_state")
    parser.add_argument("--skip-ccr", action="store_true", help="Skip CCR probe")
    args = parser.parse_args()

    print("=" * 65)
    print("TACIT — PSC + CCR Full Probe Training Pipeline")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"Window: {WINDOW_SEC}s")

    # ── Load annotations ──────────────────────────────────────────────
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

    print(f"\nLoaded: {len(raw_items)} items from {len(set(r['video_file'] for r in raw_items))} videos")

    # Build video groups
    unique_videos = sorted(set(item['video_file'] for item in raw_items))
    video_to_group = {v: i for i, v in enumerate(unique_videos)}

    # ── Run both conditions ───────────────────────────────────────────
    conditions = [
        ("base", args.base_checkpoint, "ema_encoder"),
        ("adapted", args.adapted_checkpoint, args.adapted_key),
    ]

    all_results = {}

    for cond_name, ckpt_path, ckpt_key in conditions:
        if not Path(ckpt_path).exists():
            print(f"\n  SKIP {cond_name}: not found at {ckpt_path}")
            continue

        print(f"\n{'=' * 65}")
        print(f"CONDITION: {cond_name}")
        print(f"{'=' * 65}")

        # Load encoder
        print(f"  Loading encoder from {Path(ckpt_path).name}...")
        encoder = build_encoder()
        encoder = load_checkpoint(encoder, ckpt_path, ckpt_key)
        encoder = encoder.to(DEVICE)
        encoder.eval()

        # Extract features for ALL items (once, reuse for all merge levels)
        print(f"  Extracting features for {len(raw_items)} items...")
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
        print(f"  Extracted {len(features_cache)} unique features in {time.time()-t0:.0f}s")

        # Free encoder
        del encoder
        torch.cuda.empty_cache()

        cond_results = {}

        # ── PSC at 3 granularity levels ───────────────────────────────
        merge_options = [
            ("23_class", MERGE_A),
            ("20_class", MERGE_B),
            ("10_class", MERGE_C),
        ]

        for merge_name, merge_map in merge_options:
            print(f"\n  --- PSC {merge_name} ---")

            # Apply merge
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

            print(f"    Items: {len(items)}, Classes: {len(class_names)}, Videos: {len(set(groups.tolist()))}")

            result = run_psc_groupkfold(
                features, labels, groups, class_names, cond_name,
                n_folds=N_FOLDS, n_epochs=100, lr=1e-3)

            print(f"    [{cond_name}] PSC-{merge_name}: {result['overall_acc']:.3f} "
                  f"(folds: {result['fold_accs']})")

            cond_results[f"psc_{merge_name}"] = result

        # ── CCR ───────────────────────────────────────────────────────
        if not args.skip_ccr:
            print(f"\n  --- CCR (pairwise temporal ordering) ---")
            ccr_groups = construct_ccr_groups(raw_items)
            print(f"    Groups: {len(ccr_groups)}, Total pairs: "
                  f"{sum(g['n_clips']*(g['n_clips']-1)//2 for g in ccr_groups)}")

            # Build CCR features from cache (all should already be extracted)
            filtered_ccr_groups = []
            ccr_group_features_final = []
            for gi, group in enumerate(ccr_groups):
                clip_feats = []
                for item in group['items']:
                    key = (item['video_file'], item['timestamp'])
                    if key in features_cache:
                        clip_feats.append(features_cache[key])
                if len(clip_feats) >= 4:
                    filtered_ccr_groups.append(group)
                    ccr_group_features_final.append(clip_feats)

            print(f"    CCR groups with cached features: {len(filtered_ccr_groups)}")
            
            if len(filtered_ccr_groups) >= 3:
                ccr_result = run_ccr_leave_one_out(ccr_group_features_final, filtered_ccr_groups, cond_name)
                cond_results['ccr'] = ccr_result
            else:
                print(f"    WARNING: Need ≥3 CCR groups, only have {len(filtered_ccr_groups)}")

        all_results[cond_name] = cond_results

    # ══════════════════════════════════════════════════════════════════
    # FINAL COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 75}")
    print("FINAL RESULTS — LabProc Benchmark (Organic Purification)")
    print(f"{'=' * 75}")

    # PSC comparison
    print(f"\n  PSC — Physical State Completion")
    print(f"  {'Granularity':<15} {'N_cls':>6} {'N_items':>8} ", end="")
    for cond in all_results:
        print(f"  {cond:>10}", end="")
    if len(all_results) >= 2:
        print(f"  {'delta':>8}", end="")
    print()
    print(f"  {'-'*70}")

    random_row = True
    for merge_name in ['psc_23_class', 'psc_20_class', 'psc_10_class']:
        short_name = merge_name.replace('psc_', '')
        first_result = list(all_results.values())[0].get(merge_name, {})
        n_cls = first_result.get('n_classes', 0)
        n_items = first_result.get('n_items', 0)

        # Random baseline
        if random_row and n_cls > 0:
            pass  # shown in per-row random

        print(f"  {short_name:<15} {n_cls:>6} {n_items:>8} ", end="")
        accs = []
        for cond in all_results:
            result = all_results[cond].get(merge_name, {})
            acc = result.get('overall_acc', 0)
            accs.append(acc)
            print(f"  {acc:>10.3f}", end="")
        if len(accs) >= 2:
            delta = accs[1] - accs[0]
            print(f"  {delta:>+8.3f}", end="")
        print(f"  (random: {1/n_cls:.3f})" if n_cls > 0 else "")

    # CCR comparison
    if any('ccr' in r for r in all_results.values()):
        print(f"\n  CCR — Causal Chain Reconstruction")
        print(f"  {'Metric':<20} ", end="")
        for cond in all_results:
            print(f"  {cond:>10}", end="")
        if len(all_results) >= 2:
            print(f"  {'delta':>8}", end="")
        print()
        print(f"  {'-'*55}")

        for metric, label in [('mean_tau', "Kendall's tau"), ('mean_pairwise_acc', 'Pairwise acc')]:
            print(f"  {label:<20} ", end="")
            vals = []
            for cond in all_results:
                ccr = all_results[cond].get('ccr', {})
                val = ccr.get(metric, 0)
                vals.append(val)
                print(f"  {val:>10.3f}", end="")
            if len(vals) >= 2:
                delta = vals[1] - vals[0]
                print(f"  {delta:>+8.3f}", end="")
            random_val = 0.0 if metric == 'mean_tau' else 0.5
            print(f"  (random: {random_val:.3f})")

    # Per-class breakdown for 10-class
    print(f"\n  PSC 10-class per-class breakdown:")
    print(f"  {'Class':<25} {'Count':>6} ", end="")
    for cond in all_results:
        print(f"  {cond:>8}", end="")
    if len(all_results) >= 2:
        print(f"  {'delta':>8}", end="")
    print()
    print(f"  {'-'*65}")

    merge_key = 'psc_10_class'
    first_cond = list(all_results.keys())[0]
    first_pc = all_results[first_cond].get(merge_key, {}).get('per_class', {})
    for cls in sorted(first_pc.keys()):
        count = first_pc[cls]['count']
        print(f"  {cls:<25} {count:>6} ", end="")
        accs = []
        for cond in all_results:
            pc = all_results[cond].get(merge_key, {}).get('per_class', {})
            acc = pc.get(cls, {}).get('accuracy', 0)
            accs.append(acc)
            print(f"  {acc:>8.3f}", end="")
        if len(accs) >= 2:
            delta = accs[1] - accs[0]
            print(f"  {delta:>+8.3f}", end="")
        print()

    # Save everything
    save_path = Path("/home/tacit_experiment/psc_ccr_full_results.json")
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {save_path}")

    print(f"\n{'=' * 75}")
    print("DONE")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()