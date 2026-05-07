"""
labproc_tacit/probes/ted_visual.py — TED-Visual triplet probe.

A siamese projection head (shared across anchor / correct / distractor)
trained with triplet margin loss. The probe learns to project features
into a 128-dim space where cos(anchor, correct) > cos(anchor, distractor).

Symmetric by construction (single projection applied to all three inputs)
so it cannot learn position-specific shortcuts.

Reports four numbers per condition for the same triplets:
  - siamese (trained projection + cosine ranking, the headline)
  - cos_projected (same as siamese; kept for clarity)
  - cos_raw (parameter-free baseline on raw V-JEPA features)
  - anchor_ablation (zero out the anchor at eval; should drop to ~50%)

Reports overall + per-difficulty + Strict Hard subset accuracy.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

from labproc_tacit.encoder import DEVICE


class SiameseProjection(nn.Module):
    """Single shared projection head. Output is L2-normalized so cosine = dot."""

    def __init__(self, embed_dim: int = 1024, hidden: int = 256,
                 proj_dim: int = 128):
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
        return F.normalize(z, dim=-1)


def triplet_score(proj, anchor, correct, distractor):
    """Returns (sim_correct, sim_distractor) in projected space."""
    z_anchor = proj(anchor)
    z_correct = proj(correct)
    z_distractor = proj(distractor)
    sim_correct = (z_anchor * z_correct).sum(dim=-1)
    sim_distractor = (z_anchor * z_distractor).sum(dim=-1)
    return sim_correct, sim_distractor


def triplet_margin_loss(sim_correct, sim_distractor, margin: float = 0.2):
    """Hinge: penalize whenever distractor is within `margin` of correct."""
    return F.relu(margin - sim_correct + sim_distractor).mean()


_video_to_group: dict[str, int] = {}


def _video_to_group_id(video_file: str) -> int:
    """Stable int ID per video filename, for GroupKFold."""
    if video_file not in _video_to_group:
        _video_to_group[video_file] = len(_video_to_group)
    return _video_to_group[video_file]


def train_eval_siamese_groupkfold(
    triplets_data: list[dict],
    n_folds: int = 5,
    n_epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 0.2,
    condition_name: str = "",
) -> dict | None:
    """Train + evaluate the siamese probe with GroupKFold by anchor video.

    `triplets_data` items must have keys:
      anchor_feat, correct_feat, distractor_feat (raw V-JEPA features)
      anchor_video (str, used for grouping)
      difficulty ('easy'/'medium'/'hard')
      strict_hard (bool; True iff difficulty=='hard' AND all three states equal)

    Returns four bool arrays of length len(triplets_data) for each diagnostic.
    """
    embed_dim = triplets_data[0]["anchor_feat"].shape[0]
    groups = np.array([_video_to_group_id(t["anchor_video"])
                        for t in triplets_data])
    n = len(triplets_data)

    unique_groups = len(set(groups.tolist()))
    actual_folds = min(n_folds, unique_groups)
    if actual_folds < 2:
        return None

    gkf = GroupKFold(n_splits=actual_folds)

    siamese_correct = np.zeros(n, dtype=bool)
    cos_proj_correct = np.zeros(n, dtype=bool)
    anchor_abl_correct = np.zeros(n, dtype=bool)

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(triplets_data, np.zeros(n), groups)):
        train_anchor = torch.stack(
            [triplets_data[i]["anchor_feat"] for i in train_idx]).to(DEVICE)
        train_correct = torch.stack(
            [triplets_data[i]["correct_feat"] for i in train_idx]).to(DEVICE)
        train_distractor = torch.stack(
            [triplets_data[i]["distractor_feat"] for i in train_idx]).to(DEVICE)

        val_anchor = torch.stack(
            [triplets_data[i]["anchor_feat"] for i in val_idx]).to(DEVICE)
        val_correct = torch.stack(
            [triplets_data[i]["correct_feat"] for i in val_idx]).to(DEVICE)
        val_distractor = torch.stack(
            [triplets_data[i]["distractor_feat"] for i in val_idx]).to(DEVICE)

        proj = SiameseProjection(embed_dim).to(DEVICE)
        optimizer = torch.optim.AdamW(proj.parameters(), lr=lr,
                                       weight_decay=0.01)

        best_acc = 0.0
        best_state = None
        for epoch in range(1, n_epochs + 1):
            proj.train()
            sim_c, sim_d = triplet_score(
                proj, train_anchor, train_correct, train_distractor)
            loss = triplet_margin_loss(sim_c, sim_d, margin=margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 25 == 0 or epoch == n_epochs:
                proj.eval()
                with torch.no_grad():
                    sim_c_v, sim_d_v = triplet_score(
                        proj, val_anchor, val_correct, val_distractor)
                    acc = (sim_c_v > sim_d_v).float().mean().item()
                if acc >= best_acc:
                    best_acc = acc
                    best_state = {k: v.clone()
                                  for k, v in proj.state_dict().items()}

        if best_state is None:
            continue
        proj.load_state_dict(best_state)
        proj.eval()

        with torch.no_grad():
            sim_c, sim_d = triplet_score(
                proj, val_anchor, val_correct, val_distractor)
            for k, i in enumerate(val_idx):
                siamese_correct[i] = bool(sim_c[k] > sim_d[k])
                cos_proj_correct[i] = bool(sim_c[k] > sim_d[k])

            zero_anchor = torch.zeros_like(val_anchor)
            sim_c_z, sim_d_z = triplet_score(
                proj, zero_anchor, val_correct, val_distractor)
            for k, i in enumerate(val_idx):
                anchor_abl_correct[i] = bool(sim_c_z[k] > sim_d_z[k])

        print(f"    [{condition_name}] Fold {fold+1}/{actual_folds}: "
              f"acc={best_acc:.3f}")

    raw_cos_correct = np.zeros(n, dtype=bool)
    for i, t in enumerate(triplets_data):
        a = F.normalize(t["anchor_feat"].unsqueeze(0), dim=-1)
        c = F.normalize(t["correct_feat"].unsqueeze(0), dim=-1)
        d = F.normalize(t["distractor_feat"].unsqueeze(0), dim=-1)
        raw_cos_correct[i] = bool((a * c).sum() > (a * d).sum())

    return {
        "siamese": siamese_correct,
        "cos_projected": cos_proj_correct,
        "cos_raw": raw_cos_correct,
        "anchor_ablation": anchor_abl_correct,
    }


def summarize_triplet_results(correct_arr: np.ndarray,
                               triplets_data: list[dict]) -> dict:
    """Compute overall + per-difficulty + Strict Hard accuracy from a bool array."""
    out = {"overall": round(float(correct_arr.mean()), 3),
           "n": len(triplets_data)}

    for diff in ["easy", "medium", "hard"]:
        idx = [i for i, t in enumerate(triplets_data)
               if t["difficulty"] == diff]
        if idx:
            out[diff] = {
                "accuracy": round(float(correct_arr[idx].mean()), 3),
                "n": len(idx),
            }

    strict_idx = [i for i, t in enumerate(triplets_data)
                  if t.get("strict_hard", False)]
    if strict_idx:
        out["hard_strict"] = {
            "accuracy": round(float(correct_arr[strict_idx].mean()), 3),
            "n": len(strict_idx),
        }
    soft_idx = [i for i, t in enumerate(triplets_data)
                if t["difficulty"] == "hard" and not t.get("strict_hard", False)]
    if soft_idx:
        out["hard_soft"] = {
            "accuracy": round(float(correct_arr[soft_idx].mean()), 3),
            "n": len(soft_idx),
        }
    return out
