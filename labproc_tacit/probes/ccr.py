"""
labproc_tacit/probes/ccr.py — CCR pairwise ordering probe.

Given two clip features (a, b), the probe predicts P(a comes before b) via
a 2-layer MLP on top of [a; b]. Evaluation is leave-one-video-out: for each
held-out video, train on pairs from all other videos, evaluate ordering on
the held-out video's clips.

Reports Kendall's tau and pairwise accuracy averaged across folds.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau

from labproc_tacit.encoder import DEVICE


class PairwiseOrderProbe(nn.Module):
    """Concat-based pairwise ordering probe. Output is logit for "a before b"."""

    def __init__(self, embed_dim: int = 1024):
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


def construct_ccr_groups(items: list[dict]) -> list[dict]:
    """Group PSC items by source video and dedupe consecutive same-state clips.

    Returns groups with >=4 distinct-state clips per video. Each group has
    keys `video_file`, `items` (in temporal order), and `n_clips`.
    """
    by_video = defaultdict(list)
    for item in items:
        by_video[item["video_file"]].append(item)

    groups = []
    for video_file, video_items in by_video.items():
        video_items.sort(key=lambda x: x["timestamp"])
        deduped = [video_items[0]]
        for item in video_items[1:]:
            if item["raw_label"] != deduped[-1]["raw_label"]:
                deduped.append(item)
        if len(deduped) >= 4:
            groups.append({
                "video_file": video_file,
                "items": deduped,
                "n_clips": len(deduped),
            })
    return groups


def generate_pairs(clip_features):
    """Generate all (i, j) pairs with random orientation.

    For each i < j in the temporal sequence, randomly choose to present
    the pair as (i, j) with label 1 ("a before b") or (j, i) with label 0.
    Returns three tensors: feat_a (P, D), feat_b (P, D), labels (P,).
    """
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
    return (
        torch.stack(feat_a),
        torch.stack(feat_b),
        torch.tensor(labels, dtype=torch.float),
    )


def predict_ordering(probe, clip_features):
    """Score each clip by its average "before" probability over all other
    clips, then return argsort descending. Higher score => earlier."""
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


def run_ccr_leave_one_out(
    group_features: list[list[torch.Tensor]],
    groups: list[dict],
    condition_name: str,
) -> dict:
    """Leave-one-video-out CCR evaluation.

    Args:
      group_features: list of length len(groups). Each entry is a list of
        per-clip features for that group (in temporal order).
      groups: parallel list of group dicts (must have `n_clips`).
      condition_name: human-readable label for logs.
    """
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
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3,
                                       weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(100):
            probe.train()
            logits = probe(train_fa, train_fb)
            loss = criterion(logits, train_lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        pair_acc = pair_correct / pair_total if pair_total > 0 else 0.0
        predicted_order = predict_ordering(probe, held_feats)
        tau, _ = kendalltau(true_order, predicted_order)
        if np.isnan(tau):
            tau = 0.0

        all_taus.append(tau)
        all_pair_accs.append(pair_acc)

    mean_tau = float(np.mean(all_taus))
    std_tau = float(np.std(all_taus))
    mean_pair = float(np.mean(all_pair_accs))

    print(f"    [{condition_name}] CCR: tau={mean_tau:.3f}±{std_tau:.3f}, "
          f"pair_acc={mean_pair:.3f}")

    return {
        "mean_tau": round(mean_tau, 3),
        "std_tau": round(std_tau, 3),
        "mean_pairwise_acc": round(mean_pair, 3),
        "per_group_tau": [round(float(t), 3) for t in all_taus],
        "n_groups": len(groups),
        "n_total_pairs": sum(
            g["n_clips"] * (g["n_clips"] - 1) // 2 for g in groups),
    }
