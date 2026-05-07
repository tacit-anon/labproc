"""
train_ccr_same_state.py — Same-State CCR Probe (the strict test)

Tests whether models can chronologically order clips that share the
SAME physical state label. Only motion dynamics distinguish them.

Why this is the structural test:
  - Standard CCR: clips have different state labels → Claude wins via
    state identification + textbook ordering
  - Same-state CCR: all clips share the same state → Claude on screenshots
    is provably at chance because the labels alone don't tell the order

Architecture:
  - Pairwise probe: given two clip features, predict which comes first
  - Leave-one-group-out: train on N-1 videos, test on 1
  - Aggregate pairwise predictions into full ordering via score ranking
  - Report: Kendall's tau, pairwise accuracy

Usage:
  python train_ccr_same_state.py \
    --groups-csv /home/tacit_annotation/ccr_same_state.csv \
    --video-root /home/downloads
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
from collections import defaultdict
from scipy.stats import kendalltau

sys.path.insert(0, '/home/.cache/torch/hub/facebookresearch_vjepa2_main')
sys.path.insert(0, '/home/tacit_experiment')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SEC = 4.0

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO + ENCODER
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


# ══════════════════════════════════════════════════════════════════════════════
# PAIRWISE ORDERING PROBE
# ══════════════════════════════════════════════════════════════════════════════

class PairwiseOrderProbe(nn.Module):
    """Given (feat_a, feat_b), predict 1 if a comes before b, else 0."""
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
        return self.net(torch.cat([feat_a, feat_b], dim=-1)).squeeze(-1)


def generate_pairs(clip_features):
    n = len(clip_features)
    feat_a, feat_b, labels = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                feat_a.append(clip_features[i])
                feat_b.append(clip_features[j])
                labels.append(1.0)  # i (a) comes before j (b) — correct
            else:
                feat_a.append(clip_features[j])
                feat_b.append(clip_features[i])
                labels.append(0.0)  # b actually comes first
    return torch.stack(feat_a), torch.stack(feat_b), torch.tensor(labels, dtype=torch.float)


def predict_ordering(probe, clip_features):
    """Aggregate pairwise predictions into full ordering."""
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


# ══════════════════════════════════════════════════════════════════════════════
# LEAVE-ONE-GROUP-OUT EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_leave_one_out(group_features, group_states, condition_name,
                            n_epochs=100, lr=1e-3):
    """
    Train pairwise probe on N-1 groups, test on the 1 held-out group.
    Repeat for each group.
    """
    n_groups = len(group_features)
    if n_groups < 3:
        print(f"  WARNING: only {n_groups} groups, results will be noisy")

    all_taus = []
    all_pair_accs = []
    per_group_results = []

    for held_out in range(n_groups):
        # Build training pairs from all OTHER groups
        train_fa, train_fb, train_lab = [], [], []
        for gi in range(n_groups):
            if gi == held_out:
                continue
            fa, fb, lab = generate_pairs(group_features[gi])
            train_fa.append(fa)
            train_fb.append(fb)
            train_lab.append(lab)

        train_fa = torch.cat(train_fa).to(DEVICE)
        train_fb = torch.cat(train_fb).to(DEVICE)
        train_lab = torch.cat(train_lab).to(DEVICE)

        # Train
        probe = PairwiseOrderProbe(1024).to(DEVICE)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(n_epochs):
            probe.train()
            logits = probe(train_fa, train_fb)
            loss = criterion(logits, train_lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
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
        per_group_results.append({
            'group_idx': held_out,
            'state': group_states[held_out],
            'n_clips': n_clips,
            'tau': round(tau, 3),
            'pair_acc': round(pair_acc, 3),
        })

        print(f"    [{condition_name}] Group {held_out+1}/{n_groups} "
              f"({group_states[held_out]}): tau={tau:+.3f}, pair_acc={pair_acc:.3f}")

    return {
        'mean_tau': round(float(np.mean(all_taus)), 3),
        'std_tau': round(float(np.std(all_taus)), 3),
        'mean_pair_acc': round(float(np.mean(all_pair_accs)), 3),
        'std_pair_acc': round(float(np.std(all_pair_accs)), 3),
        'per_group': per_group_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def find_video_path(video_file, video_path_hint, video_root):
    """Try various paths to find the video file on this machine."""
    # Try the hinted path first
    if video_path_hint and Path(video_path_hint).exists():
        return video_path_hint
    # Then try standard locations
    candidates = [
        Path(video_root) / "organic_purification" / "youtube" / video_file,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    found = list(Path(video_root).rglob(video_file))
    if found:
        return str(found[0])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups-csv", required=True,
                        help="CSV from annotate_ccr_with_claude.py")
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--base-checkpoint",
                        default="/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt")
    parser.add_argument("--adapted-checkpoint",
                        default="/home/tacit_experiment/adapted_v3_ema_motion_checkpoints/adapted_v3_ema_motion_epoch05.pth")
    parser.add_argument("--adapted-key", default="model_state")
    args = parser.parse_args()

    print("=" * 65)
    print("SAME-STATE CCR — Strict Temporal Ordering")
    print("=" * 65)

    # Load groups
    groups = []
    with open(args.groups_csv) as f:
        for row in csv.DictReader(f):
            video_path = find_video_path(row['video_file'],
                                          row.get('video_path'),
                                          args.video_root)
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
                'video_path': video_path,
                'state': row['state_label'],
                'timestamps': sorted(timestamps),
            })

    print(f"Loaded {len(groups)} groups with valid videos")
    state_counts = defaultdict(int)
    for g in groups:
        state_counts[g['state']] += 1
    for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
        print(f"  {state}: {count}")

    if len(groups) < 3:
        print("\nERROR: Need at least 3 groups for leave-one-out")
        return

    # Run for both V-JEPA conditions
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

        # Load encoder
        print(f"  Loading encoder...")
        encoder = build_encoder()
        encoder = load_checkpoint(encoder, ckpt_path, ckpt_key)
        encoder = encoder.to(DEVICE)
        encoder.eval()

        # Extract features for all clips
        total_clips = sum(len(g['timestamps']) for g in groups)
        print(f"  Extracting features for {total_clips} clips...")
        t0 = time.time()
        group_features = []
        group_states = []
        for gi, group in enumerate(groups):
            clip_feats = []
            for ts in group['timestamps']:
                frames = extract_frames_at_timestamp(group['video_path'], ts)
                feat = encode_video(encoder, frames)
                clip_feats.append(feat)
            group_features.append(clip_feats)
            group_states.append(group['state'])
            if (gi + 1) % 5 == 0:
                print(f"    Group {gi+1}/{len(groups)} ({time.time()-t0:.0f}s)")

        del encoder
        torch.cuda.empty_cache()

        # Leave-one-group-out evaluation
        print(f"\n  Leave-one-group-out evaluation...")
        result = evaluate_leave_one_out(group_features, group_states, cond_name)

        print(f"\n  [{cond_name}] Mean tau: {result['mean_tau']:+.3f} ± {result['std_tau']:.3f}")
        print(f"  [{cond_name}] Mean pair acc: {result['mean_pair_acc']:.3f} ± {result['std_pair_acc']:.3f}")

        all_results[cond_name] = result

    # ══════════════════════════════════════════════════════════════════
    # FINAL TABLE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 65}")
    print("SAME-STATE CCR RESULTS")
    print(f"{'=' * 65}")
    print(f"\n  Random baselines: tau=0.000, pair_acc=0.500")

    print(f"\n  {'Metric':<20} ", end="")
    for cond in all_results:
        print(f"  {cond:>10}", end="")
    if len(all_results) >= 2:
        print(f"  {'delta':>8}", end="")
    print()
    print(f"  {'-' * 60}")

    for metric, label in [('mean_tau', "Kendall's tau"),
                           ('mean_pair_acc', 'Pairwise accuracy')]:
        print(f"  {label:<20} ", end="")
        vals = []
        for cond in all_results:
            v = all_results[cond].get(metric, 0)
            vals.append(v)
            print(f"  {v:>10.3f}", end="")
        if len(vals) >= 2:
            print(f"  {vals[1] - vals[0]:>+8.3f}", end="")
        print()

    # Save
    save_path = Path("/home/tacit_experiment/ccr_same_state_results.json")
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {save_path}")

    print(f"""
  CRITICAL INTERPRETATION:
  
  Same-state CCR is the structural test where Claude-on-screenshots
  provably cannot win — all clips share the same physical state label,
  so "what state is this" reasoning is useless. Only motion dynamics
  distinguish chronological order.
  
  Expected: Claude on screenshots ≈ 0.500 (chance)
  
  If Tacit pair_acc > 0.55, you have empirical evidence that vision-only
  models capture motion signal that text-grounded models cannot access.
  This is the canonical proof of the world model thesis.
""")


if __name__ == "__main__":
    main()
