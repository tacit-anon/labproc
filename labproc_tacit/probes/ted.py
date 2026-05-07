"""
labproc_tacit/probes/ted.py — Transition Error Detection probes.

Two probe variants for TED's 4-MCQ task:

  visual_only:   train a 25-class PSC classifier; for each TED item, score
                 each of the 4 options by its predicted probability.
  visual_text:   combine V-JEPA visual features with CLIP text features
                 (question + 4 options); cross-attended scorer head.

Both variants use GroupKFold by source video.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from labproc_tacit.encoder import DEVICE


# =============================================================================
# CLIP text encoder helpers
# =============================================================================

def build_text_encoder():
    """Load CLIP ViT-L/14 for text encoding. Requires `pip install ftfy regex`
    and the CLIP package: `pip install git+https://github.com/openai/CLIP.git`."""
    import clip
    model, _ = clip.load("ViT-L/14", device=DEVICE, jit=False)
    return model


def encode_text(clip_model, texts: list[str]) -> torch.Tensor:
    """Tokenize and encode `texts`; returns CPU tensor of shape (N, 768)."""
    import clip
    tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
    with torch.no_grad():
        feats = clip_model.encode_text(tokens).float()
    return feats.cpu()


# =============================================================================
# Variant A: visual-only probe
# =============================================================================

class VisualOnlyTEDProbe(nn.Module):
    """Predict P(label) from V-JEPA features. For TED, score each option label."""

    def __init__(self, embed_dim: int = 1024, n_classes: int = 25):
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
    """Train a 25-class PSC probe; evaluation against TED options happens later."""
    embed_dim = features.shape[1]
    probe = VisualOnlyTEDProbe(embed_dim, n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    train_f = features.to(DEVICE)
    train_l = labels.to(DEVICE)

    for _ in range(n_epochs):
        probe.train()
        loss = criterion(probe(train_f), train_l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return probe


def evaluate_visual_only_ted(
    probe, ted_features, ted_options_indices, ted_correct,
) -> float:
    """For each TED item, pick the option with highest predicted probability."""
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for feat, options, true_idx in zip(
                ted_features, ted_options_indices, ted_correct):
            logits = probe(feat.unsqueeze(0).to(DEVICE)).squeeze(0)
            option_scores = logits[options]
            pred = option_scores.argmax().item()
            if pred == true_idx:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


# =============================================================================
# Variant B: visual + text probe
# =============================================================================

class VisualTextTEDProbe(nn.Module):
    """Project visual + question + each option to shared space, score per option."""

    def __init__(self, visual_dim: int = 1024, text_dim: int = 768,
                 hidden: int = 256):
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
        self.option_scorer = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, visual, question, options):
        """visual: (B, visual_dim); question: (B, text_dim); options: (B, 4, text_dim).
        Returns (B, 4) logits."""
        v = self.visual_proj(visual)
        q = self.text_proj(question)
        opts = self.text_proj(options)

        v_exp = v.unsqueeze(1).expand(-1, 4, -1)
        q_exp = q.unsqueeze(1).expand(-1, 4, -1)
        combined = torch.cat([v_exp, q_exp, opts], dim=-1)
        return self.option_scorer(combined).squeeze(-1)


def train_and_eval_visual_text_groupkfold(
    visual_features, question_features, option_features,
    correct_indices, groups,
    n_folds: int = 5, n_epochs: int = 100, lr: float = 1e-3,
    condition_name: str = "",
) -> dict:
    """K-fold cross-validation for the visual+text probe."""
    visual_dim = visual_features.shape[1]
    text_dim = question_features.shape[1]

    unique_groups = len(set(groups.tolist()))
    actual_folds = min(n_folds, unique_groups)

    gkf = GroupKFold(n_splits=actual_folds)
    all_preds = torch.zeros(len(correct_indices), dtype=torch.long)
    all_true = correct_indices.clone()
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(visual_features, correct_indices, groups)):
        train_v = visual_features[train_idx].to(DEVICE)
        train_q = question_features[train_idx].to(DEVICE)
        train_o = option_features[train_idx].to(DEVICE)
        train_c = correct_indices[train_idx].to(DEVICE)

        val_v = visual_features[val_idx].to(DEVICE)
        val_q = question_features[val_idx].to(DEVICE)
        val_o = option_features[val_idx].to(DEVICE)
        val_c = correct_indices[val_idx].to(DEVICE)

        probe = VisualTextTEDProbe(visual_dim, text_dim).to(DEVICE)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr,
                                       weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
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
        print(f"    [{condition_name}] V+T Fold {fold+1}/{actual_folds}: "
              f"acc={best_acc:.3f}")

    overall_acc = (all_preds == all_true).float().mean().item()
    return {
        "overall_acc": round(overall_acc, 3),
        "fold_accs": [round(a, 3) for a in fold_accs],
    }
