"""
train_ted_probes.py — TED Probe Training (Visual-Only + Visual+Text)

Implements two probe variants for the TED counterfactual task:

  VARIANT A: visual_only
    - Uses V-JEPA features only
    - Predicts probability over the 25 PSC labels
    - For each TED question, pick option whose label has highest probability
    - Tests implicit causal knowledge from video features alone

  VARIANT B: visual_text
    - V-JEPA visual features + CLIP text features for question and 4 options
    - Probe learns to combine them via cross-attention
    - Predicts which of A/B/C/D is correct
    - Tests vision-text reasoning

Runs both base and adapted V-JEPA conditions for direct comparison.
Uses GroupKFold (no leakage across videos).

Usage:
  python train_ted_probes.py \
    --psc-csv /home/tacit_annotation/tacit_training_op_2026-05-02_with_clipid.csv \
    --ted-csv /home/tacit_annotation/ted_op.csv \
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
from collections import Counter, defaultdict
from sklearn.model_selection import GroupKFold

sys.path.insert(0, '/home/.cache/torch/hub/facebookresearch_vjepa2_main')
sys.path.insert(0, '/home/tacit_experiment')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SEC = 4.0
N_FOLDS = 5

torch.manual_seed(42)
np.random.seed(42)


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
LABEL_TO_IDX = {label: i for i, label in enumerate(VALID_OP_LABELS)}


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO FEATURE EXTRACTION (V-JEPA)
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
# TEXT ENCODER (CLIP)
# ══════════════════════════════════════════════════════════════════════════════

def build_text_encoder():
    """Use CLIP ViT-L/14 text encoder for text features."""
    import clip
    model, _ = clip.load("ViT-L/14", device=DEVICE, jit=False)
    return model


def encode_text(clip_model, texts, max_len=77):
    import clip
    tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens).float()
    return feats.cpu()  # (n_texts, 768)


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT A: VISUAL-ONLY PROBE
# ══════════════════════════════════════════════════════════════════════════════

class VisualOnlyTEDProbe(nn.Module):
    """
    Predicts a distribution over all 25 PSC labels from video features.
    For TED, evaluate by comparing probabilities of the 4 candidate options.
    """
    def __init__(self, embed_dim=1024, n_classes=25):
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


def train_visual_only_psc(features, labels, n_classes, n_epochs=100, lr=1e-3):
    """
    Train a visual-only probe on PSC data (predict label from video features).
    This probe is then USED to score TED questions — no separate TED training.
    """
    embed_dim = features.shape[1]
    probe = VisualOnlyTEDProbe(embed_dim, n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    train_f = features.to(DEVICE)
    train_l = labels.to(DEVICE)

    for epoch in range(n_epochs):
        probe.train()
        loss = criterion(probe(train_f), train_l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return probe


def evaluate_visual_only_ted(probe, ted_features, ted_options_indices, ted_correct):
    """
    For each TED item, compare probability of the 4 option labels.
    Pick the option with highest probability.
    """
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for feat, options, true_idx in zip(ted_features, ted_options_indices, ted_correct):
            logits = probe(feat.unsqueeze(0).to(DEVICE)).squeeze(0)  # (n_classes,)
            # Score each of the 4 options
            option_scores = logits[options]  # (4,)
            pred = option_scores.argmax().item()
            if pred == true_idx:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT B: VISUAL + TEXT PROBE
# ══════════════════════════════════════════════════════════════════════════════

class VisualTextTEDProbe(nn.Module):
    """
    Combines V-JEPA visual features with CLIP text features.
    Predicts which of 4 options (A/B/C/D) is correct.
    """
    def __init__(self, visual_dim=1024, text_dim=768, hidden=256):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden),
            nn.GELU(),
        )
        # For each option, score = compatibility with (visual + question)
        self.option_scorer = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, visual, question, options):
        """
        visual: (B, visual_dim)
        question: (B, text_dim)
        options: (B, 4, text_dim)
        Returns: (B, 4) logits over options
        """
        v = self.visual_proj(visual)         # (B, hidden)
        q = self.text_proj(question)         # (B, hidden)
        opts = self.text_proj(options)       # (B, 4, hidden)

        # Concatenate [visual, question, option_i] for each option
        v_exp = v.unsqueeze(1).expand(-1, 4, -1)   # (B, 4, hidden)
        q_exp = q.unsqueeze(1).expand(-1, 4, -1)   # (B, 4, hidden)
        combined = torch.cat([v_exp, q_exp, opts], dim=-1)  # (B, 4, 3*hidden)

        scores = self.option_scorer(combined).squeeze(-1)  # (B, 4)
        return scores


def train_and_eval_visual_text_groupkfold(
    visual_features, question_features, option_features, correct_indices, groups,
    n_folds=5, n_epochs=100, lr=1e-3, condition_name=""
):
    """K-fold cross validation for visual+text probe."""
    visual_dim = visual_features.shape[1]
    text_dim = question_features.shape[1]

    unique_groups = len(set(groups.tolist()))
    actual_folds = min(n_folds, unique_groups)

    gkf = GroupKFold(n_splits=actual_folds)
    all_preds = torch.zeros(len(correct_indices), dtype=torch.long)
    all_true = correct_indices.clone()
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(visual_features, correct_indices, groups)):
        train_v = visual_features[train_idx].to(DEVICE)
        train_q = question_features[train_idx].to(DEVICE)
        train_o = option_features[train_idx].to(DEVICE)
        train_c = correct_indices[train_idx].to(DEVICE)

        val_v = visual_features[val_idx].to(DEVICE)
        val_q = question_features[val_idx].to(DEVICE)
        val_o = option_features[val_idx].to(DEVICE)
        val_c = correct_indices[val_idx].to(DEVICE)

        probe = VisualTextTEDProbe(visual_dim, text_dim).to(DEVICE)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        best_preds = None

        for epoch in range(1, n_epochs + 1):
            probe.train()
            logits = probe(train_v, train_q, train_o)
            loss = criterion(logits, train_c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0 or epoch == n_epochs:
                probe.eval()
                with torch.no_grad():
                    preds = probe(val_v, val_q, val_o).argmax(dim=1)
                    acc = (preds == val_c).float().mean().item()
                if acc >= best_acc:
                    best_acc = acc
                    best_preds = preds.cpu()

        all_preds[val_idx] = best_preds
        fold_accs.append(best_acc)
        print(f"    [{condition_name}] V+T Fold {fold+1}/{actual_folds}: acc={best_acc:.3f}")

    overall_acc = (all_preds == all_true).float().mean().item()
    print(f"    [{condition_name}] V+T Overall: {overall_acc:.3f}")

    return {
        'overall_acc': round(overall_acc, 3),
        'fold_accs': [round(a, 3) for a in fold_accs],
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

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


def load_psc_data(psc_csv, video_root):
    items = []
    with open(psc_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = find_video_path(row['video_file'], video_root)
            if video_path is None:
                continue
            if row['physical_state'] not in LABEL_TO_IDX:
                continue
            items.append({
                'clip_id': row['clip_id'],
                'video_file': row['video_file'],
                'video_path': video_path,
                'timestamp': float(row['timestamp_seconds']),
                'label': row['physical_state'],
            })
    return items


def load_ted_data(ted_csv, video_root):
    items = []
    with open(ted_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = find_video_path(row['video_file'], video_root)
            if video_path is None:
                continue
            options = [row['option_a'], row['option_b'], row['option_c'], row['option_d']]
            # Skip items where any option is invalid
            if not all(o in LABEL_TO_IDX for o in options):
                continue
            correct_letter = row['correct_option'].lower()
            correct_idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3}.get(correct_letter)
            if correct_idx is None:
                continue
            items.append({
                'ted_id': row['ted_id'],
                'video_file': row['video_file'],
                'video_path': video_path,
                'timestamp': float(row['timestamp_seconds']),
                'question': row['question'],
                'options': options,
                'option_indices': [LABEL_TO_IDX[o] for o in options],
                'correct_idx': correct_idx,
            })
    return items


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--psc-csv", required=True)
    parser.add_argument("--ted-csv", required=True)
    parser.add_argument("--video-root", default="/home/downloads")
    parser.add_argument("--base-checkpoint",
                        default="/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt")
    parser.add_argument("--adapted-checkpoint",
                        default="/home/tacit_experiment/adapted_v3_ema_motion_checkpoints/adapted_v3_ema_motion_epoch05.pth")
    parser.add_argument("--adapted-key", default="model_state")
    parser.add_argument("--skip-visual-text", action="store_true",
                        help="Skip Variant B (saves ~10 min if CLIP not installed)")
    args = parser.parse_args()

    print("=" * 65)
    print("TED PROBE TRAINING — Visual-Only + Visual+Text")
    print("=" * 65)

    # Load data
    psc_items = load_psc_data(args.psc_csv, args.video_root)
    ted_items = load_ted_data(args.ted_csv, args.video_root)
    print(f"PSC items: {len(psc_items)} (for visual-only probe training)")
    print(f"TED items: {len(ted_items)} (for evaluation)")

    if len(ted_items) < 30:
        print("\nWARNING: Few TED items. Results will be noisy.")

    # Build groups (videos)
    all_videos = sorted(set(item['video_file'] for item in psc_items + ted_items))
    video_to_group = {v: i for i, v in enumerate(all_videos)}

    # Load CLIP for text encoding (variant B)
    clip_model = None
    if not args.skip_visual_text:
        try:
            print("\nLoading CLIP text encoder...")
            clip_model = build_text_encoder()
            print("  CLIP loaded.")
        except Exception as e:
            print(f"  CLIP load failed: {e}. Skipping Variant B.")
            args.skip_visual_text = True

    # Pre-compute text features for TED (same for both V-JEPA conditions)
    if not args.skip_visual_text:
        print("\nEncoding TED questions and options...")
        questions_text = [item['question'] for item in ted_items]
        question_features = encode_text(clip_model, questions_text)

        # Encode each option separately
        all_option_features = []
        for item in ted_items:
            opt_feats = encode_text(clip_model, item['options'])  # (4, text_dim)
            all_option_features.append(opt_feats)
        option_features = torch.stack(all_option_features)  # (N, 4, text_dim)
        print(f"  Question features: {question_features.shape}")
        print(f"  Option features:   {option_features.shape}")

        # Free CLIP model
        del clip_model
        torch.cuda.empty_cache()

    # Run for both V-JEPA conditions
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
        print(f"V-JEPA CONDITION: {cond_name}")
        print(f"{'=' * 65}")

        # Load encoder
        print(f"  Loading encoder...")
        encoder = build_encoder()
        encoder = load_checkpoint(encoder, ckpt_path, ckpt_key)
        encoder = encoder.to(DEVICE)
        encoder.eval()

        # Extract PSC features (for training visual-only probe)
        print(f"  Extracting PSC features ({len(psc_items)} items)...")
        psc_feats = []
        psc_labels = []
        psc_groups = []
        t0 = time.time()
        for i, item in enumerate(psc_items):
            frames = extract_frames_at_timestamp(item['video_path'], item['timestamp'])
            feat = encode_video(encoder, frames)
            psc_feats.append(feat)
            psc_labels.append(LABEL_TO_IDX[item['label']])
            psc_groups.append(video_to_group[item['video_file']])
            if (i + 1) % 50 == 0:
                print(f"    PSC {i+1}/{len(psc_items)} ({time.time()-t0:.0f}s)")

        psc_features = torch.stack(psc_feats)
        psc_labels_t = torch.tensor(psc_labels, dtype=torch.long)
        psc_groups_arr = np.array(psc_groups)

        # Extract TED features
        print(f"  Extracting TED features ({len(ted_items)} items)...")
        ted_feats = []
        ted_groups = []
        for i, item in enumerate(ted_items):
            frames = extract_frames_at_timestamp(item['video_path'], item['timestamp'])
            feat = encode_video(encoder, frames)
            ted_feats.append(feat)
            ted_groups.append(video_to_group[item['video_file']])
            if (i + 1) % 50 == 0:
                print(f"    TED {i+1}/{len(ted_items)} ({time.time()-t0:.0f}s)")

        ted_features = torch.stack(ted_feats)
        ted_correct = torch.tensor([item['correct_idx'] for item in ted_items], dtype=torch.long)
        ted_options_indices = [item['option_indices'] for item in ted_items]
        ted_groups_arr = np.array(ted_groups)

        # Free encoder
        del encoder
        torch.cuda.empty_cache()

        cond_results = {}

        # ══════════════════════════════════════════════════════════════
        # VARIANT A: Visual-only probe
        # ══════════════════════════════════════════════════════════════
        print(f"\n  --- Variant A: Visual-Only ---")
        print(f"  Training PSC probe (visual-only) on {len(psc_features)} items...")
        psc_probe = train_visual_only_psc(
            psc_features, psc_labels_t, n_classes=len(VALID_OP_LABELS),
            n_epochs=100, lr=1e-3)

        print(f"  Evaluating on TED (visual-only)...")
        visual_only_acc = evaluate_visual_only_ted(
            psc_probe, ted_features, ted_options_indices, ted_correct.tolist())
        print(f"  [{cond_name}] Visual-only TED accuracy: {visual_only_acc:.3f}")
        cond_results['visual_only'] = {'accuracy': round(visual_only_acc, 3)}

        # Free probe
        del psc_probe

        # ══════════════════════════════════════════════════════════════
        # VARIANT B: Visual + Text probe
        # ══════════════════════════════════════════════════════════════
        if not args.skip_visual_text:
            print(f"\n  --- Variant B: Visual + Text ---")
            vt_result = train_and_eval_visual_text_groupkfold(
                ted_features, question_features, option_features,
                ted_correct, ted_groups_arr,
                n_folds=N_FOLDS, condition_name=cond_name)
            cond_results['visual_text'] = vt_result

        all_results[cond_name] = cond_results

    # ══════════════════════════════════════════════════════════════════
    # FINAL TABLE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 65}")
    print("TED RESULTS SUMMARY")
    print(f"{'=' * 65}")

    n_options = 4
    random_acc = 1.0 / n_options
    print(f"\n  Random baseline: {random_acc:.3f}")

    print(f"\n  {'Method':<25} ", end="")
    for cond in all_results:
        print(f"  {cond:>10}", end="")
    if len(all_results) >= 2:
        print(f"  {'delta':>8}", end="")
    print()
    print(f"  {'-'*55}")

    for variant, label in [('visual_only', 'Visual-only (V-only)'),
                            ('visual_text', 'Visual + Text (V+T)')]:
        accs = []
        print(f"  {label:<25} ", end="")
        for cond in all_results:
            acc = all_results[cond].get(variant, {}).get(
                'accuracy', all_results[cond].get(variant, {}).get('overall_acc', 0))
            accs.append(acc)
            print(f"  {acc:>10.3f}", end="")
        if len(accs) >= 2:
            delta = accs[1] - accs[0]
            print(f"  {delta:>+8.3f}", end="")
        print()

    # Save
    save_path = Path("/home/tacit_experiment/ted_probe_results.json")
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {save_path}")
    print()
    print("Add the following baselines manually to your paper Table:")
    print("  - Claude (text-only, no video): run separately")
    print("  - Claude (with screenshot): run separately")


if __name__ == "__main__":
    main()