"""
labproc_tacit/probes/vsd.py — Visual State Discrimination probe.

For each pair of visually-similar physical states (defined in
labproc_tacit.data.VSD_PAIRS), train a binary classifier on V-JEPA features
to discriminate state A from state B. This isolates fine-grained visual
discrimination from textual labels (the same equipment / substance / camera
angle is shared within each pair).

Cross-validation is GroupKFold by source video.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from labproc_tacit.data import VSD_PAIRS
from labproc_tacit.encoder import DEVICE


class BinaryProbe(nn.Module):
    """Two-class MLP. Used for VSD pair-level discrimination."""

    def __init__(self, embed_dim: int = 1024):
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


def _train_eval_binary_groupkfold(
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: np.ndarray,
    n_folds: int = 5,
    n_epochs: int = 100,
    lr: float = 1e-3,
    condition_name: str = "",
):
    embed_dim = features.shape[1]
    unique_groups = len(set(groups.tolist()))
    actual_folds = min(n_folds, unique_groups)
    if actual_folds < 2:
        return None

    gkf = GroupKFold(n_splits=actual_folds)
    all_preds = torch.zeros(len(labels), dtype=torch.long)
    all_true = labels.clone()
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(features, labels, groups)):
        if labels[train_idx].unique().numel() < 2:
            continue

        train_f = features[train_idx].to(DEVICE)
        train_l = labels[train_idx].to(DEVICE)
        val_f = features[val_idx].to(DEVICE)
        val_l = labels[val_idx].to(DEVICE)

        probe = BinaryProbe(embed_dim).to(DEVICE)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr,
                                       weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
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
        "overall_acc": round(overall_acc, 3),
        "mean_fold_acc": round(float(np.mean(fold_accs)), 3),
        "std_fold_acc": round(float(np.std(fold_accs)), 3),
        "fold_accs": [round(a, 3) for a in fold_accs],
        "n_items": len(labels),
        "n_folds_used": len(fold_accs),
    }


def run_vsd(
    features_cache: dict,
    psc_items: list[dict],
    video_to_group: dict[str, int],
    condition_name: str,
) -> dict | None:
    """For each pair in VSD_PAIRS, train a binary classifier and report acc.

    `features_cache` maps (video_file, timestamp) -> feature tensor.
    `video_to_group` maps video_file -> group id (int) for GroupKFold.
    """
    print(f"\n  --- VSD: Visual State Discrimination ---")

    pair_results: dict[str, dict] = {}
    for label_a, label_b, similarity, distinction in VSD_PAIRS:
        pair_items = [item for item in psc_items
                      if item["label"] in (label_a, label_b)]

        n_a = sum(1 for item in pair_items if item["label"] == label_a)
        n_b = sum(1 for item in pair_items if item["label"] == label_b)

        if n_a < 3 or n_b < 3:
            print(f"    SKIP {label_a} vs {label_b}: insufficient "
                  f"({n_a} vs {n_b})")
            continue

        feats = []
        labels_list = []
        groups_list = []
        for item in pair_items:
            key = (item["video_file"], item["timestamp"])
            if key in features_cache:
                feats.append(features_cache[key])
                labels_list.append(0 if item["label"] == label_a else 1)
                groups_list.append(video_to_group[item["video_file"]])

        if len(feats) < 6:
            continue

        features = torch.stack(feats)
        labels = torch.tensor(labels_list, dtype=torch.long)
        groups = np.array(groups_list)

        result = _train_eval_binary_groupkfold(
            features, labels, groups, n_folds=3,
            condition_name=f"{condition_name}/{label_a}_vs_{label_b}")

        if result is None:
            continue

        pair_results[f"{label_a}__vs__{label_b}"] = {
            **result,
            "n_a": n_a,
            "n_b": n_b,
            "visual_similarity": similarity,
            "visual_distinction": distinction,
        }

        print(f"    {label_a} ({n_a}) vs {label_b} ({n_b}): "
              f"acc = {result['overall_acc']:.3f} "
              f"(fold: {result['mean_fold_acc']:.3f} ± "
              f"{result['std_fold_acc']:.3f})")

    if not pair_results:
        return None

    accs = [p["overall_acc"] for p in pair_results.values()]
    mean_acc = float(np.mean(accs))
    print(f"  [{condition_name}] VSD mean across pairs: {mean_acc:.3f}")
    return {
        "mean_acc": round(mean_acc, 3),
        "n_pairs": len(pair_results),
        "per_pair": pair_results,
    }
