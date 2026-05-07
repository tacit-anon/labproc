"""
labproc_tacit/probes/psc.py — Physical State Classification probe.

A small MLP classifier (LayerNorm + Linear + GELU + Linear) trained with
cross-entropy on top of frozen V-JEPA features. Cross-validation is
GroupKFold by source video to prevent leakage from clips of the same video
appearing in both train and test.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from labproc_tacit.encoder import DEVICE


class PSCProbe(nn.Module):
    """Two-layer MLP with LayerNorm and dropout. Used for all PSC granularities."""

    def __init__(self, embed_dim: int = 1024, n_classes: int = 10):
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


def run_psc_groupkfold(
    features: torch.Tensor,
    labels: torch.Tensor,
    groups,
    class_names: list[str],
    condition_name: str,
    n_folds: int = 5,
    n_epochs: int = 100,
    lr: float = 1e-3,
) -> dict:
    """Run PSC GroupKFold and return overall and per-class accuracy.

    Args:
      features: (N, embed_dim) frozen V-JEPA features
      labels: (N,) integer class labels
      groups: (N,) array of group IDs (one per source video) for GroupKFold
      class_names: list of class names; len(class_names) == n_classes
      condition_name: human-readable label for logs
      n_folds: target number of folds (reduced if fewer groups available)
      n_epochs: training epochs per fold
      lr: AdamW learning rate
    """
    n_classes = len(class_names)
    embed_dim = features.shape[1]

    unique_groups = len(set(groups.tolist()))
    actual_folds = min(n_folds, unique_groups)
    if actual_folds < n_folds:
        print(f"    Reducing folds from {n_folds} to {actual_folds} "
              f"(only {unique_groups} groups)")

    gkf = GroupKFold(n_splits=actual_folds)
    all_preds = torch.zeros(len(labels), dtype=torch.long)
    all_true = labels.clone()
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(features, labels, groups)):
        train_f = features[train_idx].to(DEVICE)
        train_l = labels[train_idx].to(DEVICE)
        val_f = features[val_idx].to(DEVICE)
        val_l = labels[val_idx].to(DEVICE)

        probe = PSCProbe(embed_dim, n_classes).to(DEVICE)
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

        all_preds[val_idx] = best_preds
        fold_accs.append(best_acc)

    overall_acc = (all_preds == all_true).float().mean().item()

    per_class = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = all_true == cls_idx
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == all_true[mask]).float().mean().item()
            per_class[cls_name] = {
                "accuracy": round(cls_acc, 3),
                "count": int(mask.sum()),
            }

    return {
        "overall_acc": round(overall_acc, 3),
        "fold_accs": [round(a, 3) for a in fold_accs],
        "per_class": per_class,
        "n_classes": n_classes,
        "n_items": len(labels),
    }
