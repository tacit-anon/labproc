"""
labproc_tacit/probes/ccr_same_state.py — Same-State CCR (v2 pilot only).

Same architecture as the standard CCR pairwise probe (concat-MLP, BCE loss,
leave-one-group-out CV) but applied to groups of clips that share the SAME
physical state label. Only motion dynamics distinguish them, so a model
that relies on state identification + textbook ordering cannot win.

NOTE: The released Tacit checkpoint is NOT the appropriate evaluation target
for this task; the adaptation pipeline attenuates within-state temporal
coherence. See companion paper Section 6 for details. This module is provided
for users investigating alternative adaptation strategies; the v1 LabProc
release does not include this as a benchmark task.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau

from labproc_tacit.encoder import DEVICE


class PairwiseOrderProbe(nn.Module):
    """Same architecture as the standard CCR probe; renamed in this module
    for module clarity. Concat-based, predicts P(a before b)."""

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
        return self.net(torch.cat([feat_a, feat_b], dim=-1)).squeeze(-1)


def _generate_pairs(clip_features):
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


def _predict_ordering(probe, clip_features):
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


def evaluate_leave_one_out(
    group_features: list[list[torch.Tensor]],
    group_states: list[str],
    condition_name: str,
    n_epochs: int = 100,
    lr: float = 1e-3,
) -> dict:
    """Train pairwise probe on N-1 groups, evaluate on the held-out one.
    Repeat for each group; return aggregate Kendall's tau and pair accuracy.
    """
    n_groups = len(group_features)
    if n_groups < 3:
        print(f"  WARNING: only {n_groups} groups; results will be noisy")

    all_taus = []
    all_pair_accs = []
    per_group_results = []

    for held_out in range(n_groups):
        train_fa, train_fb, train_lab = [], [], []
        for gi in range(n_groups):
            if gi == held_out:
                continue
            fa, fb, lab = _generate_pairs(group_features[gi])
            train_fa.append(fa)
            train_fb.append(fb)
            train_lab.append(lab)

        train_fa = torch.cat(train_fa).to(DEVICE)
        train_fb = torch.cat(train_fb).to(DEVICE)
        train_lab = torch.cat(train_lab).to(DEVICE)

        probe = PairwiseOrderProbe(1024).to(DEVICE)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr,
                                       weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        for _ in range(n_epochs):
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

        predicted_order = _predict_ordering(probe, held_feats)
        tau, _ = kendalltau(true_order, predicted_order)
        if np.isnan(tau):
            tau = 0.0

        all_taus.append(tau)
        all_pair_accs.append(pair_acc)
        per_group_results.append({
            "group_idx": held_out,
            "state": group_states[held_out],
            "n_clips": n_clips,
            "tau": round(tau, 3),
            "pair_acc": round(pair_acc, 3),
        })

        print(f"    [{condition_name}] Group {held_out+1}/{n_groups} "
              f"({group_states[held_out]}): tau={tau:+.3f}, "
              f"pair_acc={pair_acc:.3f}")

    return {
        "mean_tau": round(float(np.mean(all_taus)), 3),
        "std_tau": round(float(np.std(all_taus)), 3),
        "mean_pair_acc": round(float(np.mean(all_pair_accs)), 3),
        "std_pair_acc": round(float(np.std(all_pair_accs)), 3),
        "per_group": per_group_results,
    }
