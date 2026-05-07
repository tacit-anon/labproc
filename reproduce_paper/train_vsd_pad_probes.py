"""
train_vsd_pad_probes.py — VSD + PAD Probe Training

Two tasks where vision-only models structurally outperform language models:

VSD — Visual State Discrimination (Without Language)
  - Pairs of physical states that share the same textual description but
    differ in visual signature (operator pose, equipment orientation, motion)
  - Binary classification within each pair
  - Claude cannot solve this from text alone
  - Uses existing PSC annotations — no new labels needed

PAD — Procedural Anomaly Detection
  - Binary: is this clip "normal procedure execution" or "off-task / anomaly"?
  - Off-task includes: pre-procedure setup, fumbling, breaks, equipment adjustment
  - Claude cannot detect "this looks abnormal" without seeing what normal looks like
  - Requires light annotation: which clips are normal vs off-task

Usage (after PAD annotations CSV is prepared):
  python train_vsd_pad_probes.py \
    --psc-csv /home/tacit_annotation/tacit_training_op_2026-05-02_with_clipid.csv \
    --pad-csv /home/tacit_annotation/pad_op.csv \
    --video-root /home/downloads
"""

import argparse
import csv
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupKFold

sys.path.insert(0, '/home/.cache/torch/hub/facebookresearch_vjepa2_main')
sys.path.insert(0, '/home/tacit_experiment')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SEC = 4.0
N_FOLDS = 5

torch.manual_seed(42)
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# VSD CONFUSION PAIRS — visually similar states with same textual description
# ══════════════════════════════════════════════════════════════════════════════
#
# These pairs share the same equipment, same substances, often same camera angle.
# Discrimination requires fine-grained visual cues that text cannot capture.
#
# Format: (label_A, label_B, why_visually_similar, what_distinguishes_them)

VSD_PAIRS = [
    (
        'lle_two_phase_settled', 'lle_draining_lower_layer',
        'separatory funnel containing two visible layers',
        'operator hand on stopcock + layer level dropping'
    ),
    (
        'column_packed', 'column_equilibrated',
        'glass column with white silica',
        'liquid flowing through column + dripping into receiving flask'
    ),
    (
        'mixture_dissolved_hot', 'reflux_running',
        'flask on hot plate with clear/colored liquid',
        'condenser presence + boiling vapor + temperature stability'
    ),
    (
        'fractions_collecting', 'fraction_analysis',
        'small vials/test tubes with collected liquid',
        'active dripping into tubes (collecting) vs static analysis under UV'
    ),
    (
        'crystals_forming', 'crystals_complete',
        'flask with white solid material',
        'turbidity dynamics (forming) vs settled crystals (complete)'
    ),
    (
        'distillation_setup_running', 'reflux_running',
        'flask with condenser, heat applied',
        'collection flask filling (distillation) vs liquid returning to flask (reflux)'
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# PAD ANNOTATION CSV FORMAT (you create this manually)
#
# pad_op.csv columns:
#   clip_id, video_file, timestamp_seconds, label
#   - label is one of: "normal_procedure", "anomaly"
#   - "normal_procedure": active procedure execution (subset of PSC items)
#   - "anomaly": off-task moments — break, fumble, talking to camera,
#                equipment adjustment between procedures, pre-procedure setup,
#                operator looking at notes, dropping equipment, walking away
#
# To create this CSV:
#   1. Take 30-50 of your PSC items as "normal_procedure" examples
#   2. Manually scrub through the same 38 videos and find 30-50 off-task moments
#   3. Label them "anomaly"
# ══════════════════════════════════════════════════════════════════════════════

VALID_OP_LABELS = [
    "mixture_crude_unreacted", "mixture_dissolved_hot",
    "crystals_forming", "crystals_complete",
    "tlc_plate_dry", "tlc_plate_spotted", "tlc_running", "tlc_developed",
    "column_dry", "column_packed", "column_equilibrated",
    "sample_loaded", "fractions_collecting", "fraction_analysis",
    "lle_two_phase_settled", "lle_draining_lower_layer",
    "distillation_setup_running", "distillate_collecting", "reflux_running",
    "rotovap_running",
    "vacuum_filtration_general", "gravity_filtration_hot",
    "analytical_weighing", "solvent_dispensing", "titration_running",
]


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

import cv2

def extract_frames_at_timestamp(video_path, timestamp_sec,
                                 n_frames=16, frame_size=384, window_sec=WINDOW_SEC):
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


def find_video_path(video_file, video_root):
    candidates = [
        Path(video_root) / "organic_purification" / "youtube" / video_file,
        Path(video_root) / "organic_purification" / "youtube" / video_file.replace('good_', ''),
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
# BINARY PROBE
# ══════════════════════════════════════════════════════════════════════════════

class BinaryProbe(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.probe = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.probe(x)


def train_eval_binary_groupkfold(features, labels, groups, n_folds=5,
                                   n_epochs=100, lr=1e-3, condition_name=""):
    """Binary classification with GroupKFold."""
    embed_dim = features.shape[1]
    unique_groups = len(set(groups.tolist()))
    actual_folds = min(n_folds, unique_groups)

    if actual_folds < 2:
        return None

    gkf = GroupKFold(n_splits=actual_folds)
    all_preds = torch.zeros(len(labels), dtype=torch.long)
    all_true = labels.clone()
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(features, labels, groups)):
        if labels[train_idx].unique().numel() < 2:
            continue  # need both classes

        train_f = features[train_idx].to(DEVICE)
        train_l = labels[train_idx].to(DEVICE)
        val_f = features[val_idx].to(DEVICE)
        val_l = labels[val_idx].to(DEVICE)

        probe = BinaryProbe(embed_dim).to(DEVICE)
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

        if best_preds is not None:
            all_preds[val_idx] = best_preds
            fold_accs.append(best_acc)

    if not fold_accs:
        return None

    overall_acc = (all_preds == all_true).float().mean().item()
    return {
        'overall_acc': round(overall_acc, 3),
        'mean_fold_acc': round(float(np.mean(fold_accs)), 3),
        'std_fold_acc': round(float(np.std(fold_accs)), 3),
        'fold_accs': [round(a, 3) for a in fold_accs],
        'n_items': len(labels),
        'n_folds_used': len(fold_accs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# VSD: VISUAL STATE DISCRIMINATION
# ══════════════════════════════════════════════════════════════════════════════

def run_vsd(features_cache, psc_items, video_to_group, condition_name):
    """For each VSD pair, train a binary classifier on label_A vs label_B."""
    print(f"\n  --- VSD: Visual State Discrimination ---")

    pair_results = {}
    for label_a, label_b, similarity, distinction in VSD_PAIRS:
        # Filter items
        pair_items = [item for item in psc_items
                      if item['label'] in (label_a, label_b)]

        n_a = sum(1 for item in pair_items if item['label'] == label_a)
        n_b = sum(1 for item in pair_items if item['label'] == label_b)

        if n_a < 3 or n_b < 3:
            print(f"    SKIP {label_a} vs {label_b}: insufficient ({n_a} vs {n_b})")
            continue

        feats = []
        labels_list = []
        groups_list = []
        for item in pair_items:
            key = (item['video_file'], item['timestamp'])
            if key in features_cache:
                feats.append(features_cache[key])
                labels_list.append(0 if item['label'] == label_a else 1)
                groups_list.append(video_to_group[item['video_file']])

        if len(feats) < 6:
            continue

        features = torch.stack(feats)
        labels = torch.tensor(labels_list, dtype=torch.long)
        groups = np.array(groups_list)

        result = train_eval_binary_groupkfold(
            features, labels, groups, n_folds=3,
            condition_name=f"{condition_name}/{label_a}_vs_{label_b}")

        if result is None:
            print(f"    SKIP {label_a} vs {label_b}: not enough fold variety")
            continue

        pair_results[f"{label_a}__vs__{label_b}"] = {
            **result,
            'n_a': n_a,
            'n_b': n_b,
            'visual_similarity': similarity,
            'visual_distinction': distinction,
        }

        print(f"    {label_a} ({n_a}) vs {label_b} ({n_b}): "
              f"acc = {result['overall_acc']:.3f} "
              f"(mean fold: {result['mean_fold_acc']:.3f} ± {result['std_fold_acc']:.3f})")

    # Aggregate
    if pair_results:
        accs = [p['overall_acc'] for p in pair_results.values()]
        mean_acc = np.mean(accs)
        print(f"  [{condition_name}] VSD mean across pairs: {mean_acc:.3f}")
        return {
            'mean_acc': round(float(mean_acc), 3),
            'n_pairs': len(pair_results),
            'per_pair': pair_results,
        }
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PAD: PROCEDURAL ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def load_pad_data(pad_csv, video_root):
    items = []
    with open(pad_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = find_video_path(row['video_file'], video_root)
            if video_path is None:
                continue
            label = row['label'].strip().lower()
            if label not in ('normal_procedure', 'anomaly'):
                continue
            items.append({
                'clip_id': row.get('clip_id', f"pad_{len(items)}"),
                'video_file': row['video_file'],
                'video_path': video_path,
                'timestamp': float(row['timestamp_seconds']),
                'label': label,
            })
    return items


def run_pad(features_cache, pad_items, video_to_group, condition_name):
    """Binary: normal_procedure vs anomaly."""
    print(f"\n  --- PAD: Procedural Anomaly Detection ---")
    print(f"  PAD items: {len(pad_items)}")

    n_normal = sum(1 for item in pad_items if item['label'] == 'normal_procedure')
    n_anomaly = sum(1 for item in pad_items if item['label'] == 'anomaly')
    print(f"    Normal: {n_normal}, Anomaly: {n_anomaly}")

    if n_normal < 5 or n_anomaly < 5:
        print(f"  SKIP: insufficient examples")
        return None

    feats = []
    labels_list = []
    groups_list = []
    for item in pad_items:
        key = (item['video_file'], item['timestamp'])
        if key in features_cache:
            feats.append(features_cache[key])
            labels_list.append(0 if item['label'] == 'normal_procedure' else 1)
            groups_list.append(video_to_group[item['video_file']])

    if len(feats) < 10:
        return None

    features = torch.stack(feats)
    labels = torch.tensor(labels_list, dtype=torch.long)
    groups = np.array(groups_list)

    result = train_eval_binary_groupkfold(
        features, labels, groups, n_folds=N_FOLDS, condition_name=condition_name)

    if result:
        print(f"  [{condition_name}] PAD overall: {result['overall_acc']:.3f}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def load_psc_data(psc_csv, video_root):
    items = []
    with open(psc_csv) as f:
        for row in csv.DictReader(f):
            video_path = find_video_path(row['video_file'], video_root)
            if video_path is None:
                continue
            if row['physical_state'] not in VALID_OP_LABELS:
                continue
            items.append({
                'video_file': row['video_file'],
                'video_path': video_path,
                'timestamp': float(row['timestamp_seconds']),
                'label': row['physical_state'],
            })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psc-csv", required=True)
    parser.add_argument("--pad-csv", default=None,
                        help="PAD annotations CSV (optional - skip PAD if not provided)")
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--base-checkpoint",
                        default="/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt")
    parser.add_argument("--adapted-checkpoint",
                        default="/home/tacit_experiment/adapted_v3_ema_motion_checkpoints/adapted_v3_ema_motion_epoch05.pth")
    parser.add_argument("--adapted-key", default="model_state")
    args = parser.parse_args()

    print("=" * 65)
    print("VSD + PAD PROBE TRAINING (Vision-Only Advantage Tasks)")
    print("=" * 65)

    psc_items = load_psc_data(args.psc_csv, args.video_root)
    print(f"PSC items: {len(psc_items)}")

    pad_items = []
    if args.pad_csv and Path(args.pad_csv).exists():
        pad_items = load_pad_data(args.pad_csv, args.video_root)
        print(f"PAD items: {len(pad_items)}")
    else:
        print(f"PAD CSV not provided — VSD only")

    # Build video groups
    all_videos = sorted(set(item['video_file'] for item in psc_items + pad_items))
    video_to_group = {v: i for i, v in enumerate(all_videos)}

    # Run for both conditions
    conditions = [
        ("base", args.base_checkpoint, "ema_encoder"),
        ("adapted", args.adapted_checkpoint, args.adapted_key),
    ]

    all_results = {}

    for cond_name, ckpt_path, ckpt_key in conditions:
        if not Path(ckpt_path).exists():
            print(f"\n  SKIP {cond_name}")
            continue

        print(f"\n{'=' * 65}")
        print(f"V-JEPA CONDITION: {cond_name}")
        print(f"{'=' * 65}")

        encoder = build_encoder()
        encoder = load_checkpoint(encoder, ckpt_path, ckpt_key)
        encoder = encoder.to(DEVICE)
        encoder.eval()

        # Extract features for ALL items (PSC + PAD)
        all_items = psc_items + pad_items
        unique_keys = set()
        for item in all_items:
            unique_keys.add((item['video_file'], item['timestamp']))

        print(f"  Extracting features for {len(unique_keys)} unique clips...")
        features_cache = {}
        t0 = time.time()
        for i, item in enumerate(all_items):
            key = (item['video_file'], item['timestamp'])
            if key not in features_cache:
                frames = extract_frames_at_timestamp(item['video_path'], item['timestamp'])
                features_cache[key] = encode_video(encoder, frames)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(all_items)} ({time.time()-t0:.0f}s)")

        del encoder
        torch.cuda.empty_cache()

        cond_results = {}

        # VSD
        vsd_result = run_vsd(features_cache, psc_items, video_to_group, cond_name)
        if vsd_result:
            cond_results['vsd'] = vsd_result

        # PAD (if available)
        if pad_items:
            pad_result = run_pad(features_cache, pad_items, video_to_group, cond_name)
            if pad_result:
                cond_results['pad'] = pad_result

        all_results[cond_name] = cond_results

    # ══════════════════════════════════════════════════════════════════
    # FINAL TABLE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 65}")
    print("VSD + PAD RESULTS SUMMARY")
    print(f"{'=' * 65}")

    if not all_results:
        print("No results.")
        return

    # VSD summary
    print(f"\n  VSD — Visual State Discrimination (binary, random=0.500)")
    print(f"  {'Pair':<55} ", end="")
    for cond in all_results:
        print(f"  {cond:>10}", end="")
    print()
    print(f"  {'-'*80}")

    # Get all pair keys
    all_pairs = set()
    for cond_results in all_results.values():
        if 'vsd' in cond_results:
            all_pairs.update(cond_results['vsd']['per_pair'].keys())

    for pair in sorted(all_pairs):
        print(f"  {pair[:55]:<55} ", end="")
        for cond in all_results:
            vsd = all_results[cond].get('vsd', {})
            pair_data = vsd.get('per_pair', {}).get(pair, {})
            acc = pair_data.get('overall_acc', None)
            if acc is not None:
                print(f"  {acc:>10.3f}", end="")
            else:
                print(f"  {'-':>10}", end="")
        print()

    print(f"  {'-'*80}")
    print(f"  {'MEAN':<55} ", end="")
    for cond in all_results:
        mean = all_results[cond].get('vsd', {}).get('mean_acc', None)
        if mean is not None:
            print(f"  {mean:>10.3f}", end="")
        else:
            print(f"  {'-':>10}", end="")
    print()

    # PAD summary
    if any('pad' in r for r in all_results.values()):
        print(f"\n  PAD — Procedural Anomaly Detection (binary, random=0.500)")
        print(f"  {'Metric':<25} ", end="")
        for cond in all_results:
            print(f"  {cond:>10}", end="")
        print()
        print(f"  {'-'*55}")

        for metric_key, metric_label in [('overall_acc', 'Overall accuracy'),
                                          ('mean_fold_acc', 'Mean fold acc'),
                                          ('std_fold_acc', 'Std fold acc')]:
            print(f"  {metric_label:<25} ", end="")
            for cond in all_results:
                pad = all_results[cond].get('pad', {})
                val = pad.get(metric_key, None)
                if val is not None:
                    print(f"  {val:>10.3f}", end="")
                else:
                    print(f"  {'-':>10}", end="")
            print()

    # Save
    save_path = Path("/home/tacit_experiment/vsd_pad_results.json")
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {save_path}")

    # Key takeaway message
    print(f"\n{'=' * 65}")
    print("INTERPRETATION")
    print(f"{'=' * 65}")
    print("""
  These tasks isolate vision-only signals:
    - VSD: same textual description, different visual states
    - PAD: out-of-distribution detection from visual statistics

  Claude (image-language) cannot solve these from text alone.
  If V-JEPA improvements > Claude's performance here, that is direct
  evidence of a vision-only competitive advantage.
""")


if __name__ == "__main__":
    main()