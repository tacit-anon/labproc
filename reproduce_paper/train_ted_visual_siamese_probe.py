"""
train_ted_visual_siamese_probe.py — Siamese Probe for TED-Visual

DROP-IN REPLACEMENT for train_ted_visual_probe.py with three architectural fixes:

  FIX 1: Siamese projection head (shared weights across anchor/correct/distractor).
         Symmetric by construction — eliminates the candidate-position shortcut
         that the concatenation probe is vulnerable to.

  FIX 2: Triplet ranking loss instead of classification.
         Trains the projection so that cos(g(anchor), g(correct)) >
         cos(g(anchor), g(distractor)) by a margin. The model learns a
         task-relevant similarity metric instead of memorizing positional
         patterns from a 3072-dim concatenated input.

  FIX 3: Drastically fewer parameters (~270K vs ~1.6M in concat probe).
         Single projection head: 1024 → 256 → 128 with LayerNorm.
         Better-suited to the ~50-150 examples per fold regime.

Reports four numbers per condition (base / adapted):
  - Siamese probe (trained, the headline number)
  - Cosine baseline in projected space (uses trained projection, no decision layer)
  - Raw cosine baseline (parameter-free, on raw V-JEPA features — for comparison)
  - Anchor-ablation diagnostic (zeros out anchor at eval; should drop to chance
    if the probe is actually using anchor information)

Also reports STRICT HARD subset separately when anchor_state, correct_state,
distractor_state columns are present in the CSV — this is the load-bearing
result for the paper.

Usage:
  python train_ted_visual_siamese_probe.py \
    --triplets-csv /home/tacit_annotation/ted_visual.csv \
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
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import GroupKFold

sys.path.insert(0, '/home/.cache/torch/hub/facebookresearch_vjepa2_main')
sys.path.insert(0, '/home/tacit_experiment')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SEC = 4.0

torch.manual_seed(42)
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO + ENCODER (unchanged from original)
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
    clean = video_file.replace('good_', '').replace('ok_', '').replace('bad_', '')
    found = list(Path(video_root).rglob(clean))
    if found:
        return str(found[0])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SIAMESE PROBE
# ══════════════════════════════════════════════════════════════════════════════

class SiameseProjection(nn.Module):
    """
    Single shared projection head applied to anchor, correct, and distractor
    independently. This is the *only* learned parameter in the probe.

    Symmetric by construction: cannot learn position-specific shortcuts because
    the same head processes all three inputs.

    ~270K parameters total (vs ~1.6M in the concat probe).
    """
    def __init__(self, embed_dim=1024, hidden=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)  # unit-norm so cosine == dot product


def triplet_score(proj, anchor, correct, distractor):
    """
    Returns (sim_correct, sim_distractor) after passing all three through
    the shared projection head. Both are cosine similarities to the projected
    anchor.
    """
    z_anchor = proj(anchor)
    z_correct = proj(correct)
    z_distractor = proj(distractor)
    sim_correct = (z_anchor * z_correct).sum(dim=-1)
    sim_distractor = (z_anchor * z_distractor).sum(dim=-1)
    return sim_correct, sim_distractor


def triplet_margin_loss(sim_correct, sim_distractor, margin=0.2):
    """
    Hinge loss: penalize whenever distractor similarity is within `margin` of
    correct similarity. Pushes correct above distractor in projected space.
    """
    return F.relu(margin - sim_correct + sim_distractor).mean()


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING + EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_eval_siamese_groupkfold(triplets_data, n_folds=5, n_epochs=200,
                                   lr=1e-3, margin=0.2, condition_name=""):
    """
    triplets_data fields needed:
      anchor_feat, correct_feat, distractor_feat (raw V-JEPA features)
      anchor_video (for GroupKFold)
      difficulty, strict_hard (bool, optional — for subset reporting)

    Returns dict with overall, per-difficulty, and strict-Hard accuracies for:
      - siamese (trained projection + cosine ranking)
      - cosine_projected (use trained projection but no decision layer; sanity)
      - cosine_raw (parameter-free baseline on raw features)
      - anchor_ablation (zero anchor at eval — should drop to ~50% if probe works)
    """
    embed_dim = triplets_data[0]['anchor_feat'].shape[0]
    groups = np.array([video_to_group_id(t['anchor_video']) for t in triplets_data])
    n = len(triplets_data)

    unique_groups = len(set(groups.tolist()))
    actual_folds = min(n_folds, unique_groups)
    if actual_folds < 2:
        return None

    gkf = GroupKFold(n_splits=actual_folds)

    # Per-example correctness flags (1 if probe ranked correct > distractor)
    siamese_correct = np.zeros(n, dtype=bool)
    cos_proj_correct = np.zeros(n, dtype=bool)
    anchor_abl_correct = np.zeros(n, dtype=bool)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(triplets_data, np.zeros(n), groups)):
        train_anchor = torch.stack([triplets_data[i]['anchor_feat'] for i in train_idx]).to(DEVICE)
        train_correct = torch.stack([triplets_data[i]['correct_feat'] for i in train_idx]).to(DEVICE)
        train_distractor = torch.stack([triplets_data[i]['distractor_feat'] for i in train_idx]).to(DEVICE)

        val_anchor = torch.stack([triplets_data[i]['anchor_feat'] for i in val_idx]).to(DEVICE)
        val_correct = torch.stack([triplets_data[i]['correct_feat'] for i in val_idx]).to(DEVICE)
        val_distractor = torch.stack([triplets_data[i]['distractor_feat'] for i in val_idx]).to(DEVICE)

        proj = SiameseProjection(embed_dim).to(DEVICE)
        optimizer = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=0.01)

        best_acc = 0
        best_state = None

        for epoch in range(1, n_epochs + 1):
            proj.train()
            sim_c, sim_d = triplet_score(proj, train_anchor, train_correct, train_distractor)
            loss = triplet_margin_loss(sim_c, sim_d, margin=margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 25 == 0 or epoch == n_epochs:
                proj.eval()
                with torch.no_grad():
                    sim_c_v, sim_d_v = triplet_score(proj, val_anchor, val_correct, val_distractor)
                    acc = (sim_c_v > sim_d_v).float().mean().item()
                if acc >= best_acc:
                    best_acc = acc
                    best_state = {k: v.clone() for k, v in proj.state_dict().items()}

        if best_state is None:
            continue
        proj.load_state_dict(best_state)
        proj.eval()
        fold_accs.append(best_acc)

        with torch.no_grad():
            # Siamese (trained, headline)
            sim_c, sim_d = triplet_score(proj, val_anchor, val_correct, val_distractor)
            for k, i in enumerate(val_idx):
                siamese_correct[i] = bool(sim_c[k] > sim_d[k])

            # Cosine in projected space (same as siamese here; kept for clarity)
            for k, i in enumerate(val_idx):
                cos_proj_correct[i] = bool(sim_c[k] > sim_d[k])

            # Anchor ablation: zero out anchor → does the probe collapse to chance?
            zero_anchor = torch.zeros_like(val_anchor)
            sim_c_z, sim_d_z = triplet_score(proj, zero_anchor, val_correct, val_distractor)
            for k, i in enumerate(val_idx):
                anchor_abl_correct[i] = bool(sim_c_z[k] > sim_d_z[k])

        print(f"    [{condition_name}] Siamese Fold {fold+1}/{actual_folds}: acc={best_acc:.3f}")

    # Raw cosine baseline (no probe at all)
    raw_cos_correct = np.zeros(n, dtype=bool)
    for i, t in enumerate(triplets_data):
        a = F.normalize(t['anchor_feat'].unsqueeze(0), dim=-1)
        c = F.normalize(t['correct_feat'].unsqueeze(0), dim=-1)
        d = F.normalize(t['distractor_feat'].unsqueeze(0), dim=-1)
        raw_cos_correct[i] = bool((a * c).sum() > (a * d).sum())

    return {
        'siamese': siamese_correct,
        'cos_projected': cos_proj_correct,
        'cos_raw': raw_cos_correct,
        'anchor_ablation': anchor_abl_correct,
    }


def summarize(correct_arr, triplets_data):
    """Compute overall, per-difficulty, and strict-Hard accuracy from a bool array."""
    n = len(triplets_data)
    out = {'overall': round(float(correct_arr.mean()), 3), 'n': n}

    for diff in ['easy', 'medium', 'hard']:
        idx = [i for i, t in enumerate(triplets_data) if t['difficulty'] == diff]
        if idx:
            out[diff] = {
                'accuracy': round(float(correct_arr[idx].mean()), 3),
                'n': len(idx),
            }

    # Strict Hard: anchor_state == correct_state == distractor_state
    strict_idx = [i for i, t in enumerate(triplets_data)
                  if t.get('strict_hard', False)]
    if strict_idx:
        out['hard_strict'] = {
            'accuracy': round(float(correct_arr[strict_idx].mean()), 3),
            'n': len(strict_idx),
        }
    soft_idx = [i for i, t in enumerate(triplets_data)
                if t['difficulty'] == 'hard' and not t.get('strict_hard', False)]
    if soft_idx:
        out['hard_soft'] = {
            'accuracy': round(float(correct_arr[soft_idx].mean()), 3),
            'n': len(soft_idx),
        }
    return out


_video_to_group = {}
def video_to_group_id(video_file):
    if video_file not in _video_to_group:
        _video_to_group[video_file] = len(_video_to_group)
    return _video_to_group[video_file]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplets-csv", required=True)
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--base-checkpoint",
                        default="/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt")
    parser.add_argument("--adapted-checkpoint",
                        default="/home/tacit_experiment/adapted_v3_ema_motion_checkpoints/adapted_v3_ema_motion_epoch05.pth")
    parser.add_argument("--adapted-key", default="model_state")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of seeds to average over (variance check)")
    parser.add_argument("--margin", type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 70)
    print("TED-VISUAL — V-JEPA SIAMESE PROBE")
    print("=" * 70)

    # Load triplets — note: now using correct/distractor directly, not A/B
    triplets_raw = []
    with open(args.triplets_csv) as f:
        for row in csv.DictReader(f):
            anchor_path = find_video_path(row['anchor_video'], args.video_root)
            correct_path = find_video_path(row['correct_video'], args.video_root)
            distractor_path = find_video_path(row['distractor_video'], args.video_root)
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

    print(f"Loaded {len(triplets_raw)} triplets")
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

    if len(triplets_raw) < 30:
        print("\nERROR: Need at least 30 triplets")
        return

    conditions = [
        ("base", args.base_checkpoint, "ema_encoder"),
        ("adapted", args.adapted_checkpoint, args.adapted_key),
    ]

    all_results = {}

    for cond_name, ckpt_path, ckpt_key in conditions:
        if not Path(ckpt_path).exists():
            print(f"\n  SKIP {cond_name}: checkpoint not found")
            continue

        print(f"\n{'=' * 70}")
        print(f"V-JEPA CONDITION: {cond_name}")
        print(f"{'=' * 70}")

        print(f"  Loading encoder from {Path(ckpt_path).name}...")
        encoder = build_encoder()
        encoder = load_checkpoint(encoder, ckpt_path, ckpt_key)
        encoder = encoder.to(DEVICE)
        encoder.eval()

        unique_clips = set()
        for t in triplets_raw:
            unique_clips.add((t['anchor_path'], t['anchor_ts']))
            unique_clips.add((t['correct_path'], t['correct_ts']))
            unique_clips.add((t['distractor_path'], t['distractor_ts']))

        print(f"  Extracting features for {len(unique_clips)} unique clips...")
        features_cache = {}
        t0 = time.time()
        for i, (path, ts) in enumerate(unique_clips):
            frames = extract_frames_at_timestamp(path, ts)
            features_cache[(path, ts)] = encode_video(encoder, frames)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(unique_clips)} ({time.time()-t0:.0f}s)")

        del encoder
        torch.cuda.empty_cache()

        triplets_data = []
        for t in triplets_raw:
            triplets_data.append({
                **t,
                'anchor_feat': features_cache[(t['anchor_path'], t['anchor_ts'])],
                'correct_feat': features_cache[(t['correct_path'], t['correct_ts'])],
                'distractor_feat': features_cache[(t['distractor_path'], t['distractor_ts'])],
            })

        # Multi-seed run for variance check
        print(f"\n  --- Siamese probe (5-fold GroupKFold × {args.n_seeds} seeds) ---")
        seed_results = []
        for seed in range(args.n_seeds):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)
            print(f"\n  Seed {seed + 1}/{args.n_seeds}")
            res = train_eval_siamese_groupkfold(
                triplets_data, n_folds=5, margin=args.margin,
                condition_name=f"{cond_name}-s{seed}",
            )
            if res is not None:
                seed_results.append(res)

        # Aggregate seeds: mean + std of overall accuracy
        cond_results = {}
        for probe_name in ['siamese', 'cos_projected', 'cos_raw', 'anchor_ablation']:
            per_seed_acc = [r[probe_name].mean() for r in seed_results]
            mean_arr = np.mean([r[probe_name] for r in seed_results], axis=0) > 0.5
            cond_results[probe_name] = summarize(mean_arr, triplets_data)
            cond_results[probe_name]['overall_mean'] = round(float(np.mean(per_seed_acc)), 3)
            cond_results[probe_name]['overall_std'] = round(float(np.std(per_seed_acc)), 3)

        all_results[cond_name] = cond_results

        # Quick print
        s = cond_results['siamese']
        a = cond_results['anchor_ablation']
        print(f"\n  [{cond_name}] Siamese:        {s['overall_mean']:.3f} ± {s['overall_std']:.3f}")
        print(f"  [{cond_name}] Anchor ablation: {a['overall_mean']:.3f} ± {a['overall_std']:.3f}  "
              f"({'OK — probe uses anchor' if a['overall_mean'] < s['overall_mean'] - 0.05 else 'WARNING: probe ignoring anchor'})")
        if 'hard_strict' in s:
            print(f"  [{cond_name}] Strict Hard:     {s['hard_strict']['accuracy']:.3f} (n={s['hard_strict']['n']})")

    # ══════════════════════════════════════════════════════════════════
    # FINAL TABLE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 80}")
    print("TED-VISUAL — SIAMESE PROBE RESULTS")
    print(f"{'=' * 80}")
    print(f"\n  Random baseline: 0.500")
    print(f"  Anchor-ablation should be near 0.500 for a working probe.\n")

    def print_block(probe_name, label):
        print(f"  --- {label} ---")
        print(f"  {'Subset':<14} ", end="")
        for cond in all_results:
            print(f"  {cond:>14}", end="")
        if len(all_results) >= 2:
            print(f"  {'Δ (adapted-base)':>18}", end="")
        print()
        print(f"  {'-' * 70}")

        for subset in ['easy', 'medium', 'hard', 'hard_strict', 'hard_soft', 'overall']:
            if subset == 'overall':
                row_label = 'OVERALL'
                getter = lambda r: (r.get('overall_mean'), r.get('overall_std'))
            else:
                row_label = subset
                getter = lambda r: (r.get(subset, {}).get('accuracy'), None)
            vals = []
            print(f"  {row_label:<14} ", end="")
            for cond in all_results:
                r = all_results[cond].get(probe_name, {})
                v, s = getter(r)
                vals.append(v)
                if v is None:
                    print(f"  {'-':>14}", end="")
                elif s is not None:
                    print(f"  {v:>7.3f}±{s:.3f}", end="")
                else:
                    n = r.get(subset, {}).get('n', '')
                    print(f"  {v:>7.3f} (n={n:<3})" if n else f"  {v:>14.3f}", end="")
            if len(vals) >= 2 and None not in vals:
                print(f"  {vals[1]-vals[0]:>+18.3f}", end="")
            print()
        print()

    print_block('siamese',         'SIAMESE PROBE (trained, headline result)')
    print_block('cos_raw',         'RAW COSINE (parameter-free, raw V-JEPA features)')
    print_block('anchor_ablation', 'ANCHOR-ABLATION DIAGNOSTIC (zero anchor at eval)')

    save_path = Path("/home/tacit_experiment/ted_visual_siamese_results.json")
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to: {save_path}")

    print(f"\n{'=' * 80}")
    print("INTERPRETATION GUIDE")
    print(f"{'=' * 80}")
    print("""
  The siamese probe is the headline. Compare to anchor-ablation:
    siamese > anchor_ablation by >5pp → probe is meaningfully using anchor info
    siamese ≈ anchor_ablation         → probe ignoring anchor (data too small / probe broken)

  The hard_strict row is what differentiates this from VLM evaluation:
    - VLMs on screenshots score ~50% on this subset (Claude: 54.5%)
    - V-JEPA must score meaningfully > 50% here, with adapted > base, to claim
      that lab-trained video representations capture motion signal that
      vision-language models structurally cannot access.

  Raw cosine baseline tells you whether the trained probe is doing real work:
    - If raw cosine ≈ siamese, V-JEPA's latent geometry already separates
      correct from distractor; the probe just polishes it.
    - If raw cosine << siamese, the trained projection is essential.
""")


if __name__ == "__main__":
    main()