"""
adapt_v3.py — Domain-Adaptive Continued Pretraining of V-JEPA 2.1 (v3)

Two key improvements over v1/v2:

1. EMA TARGET ENCODER (prevents representation collapse)
   In v1/v2, the same encoder was used to produce both the prediction
   and the target. The model found a shortcut: make all representations
   identical (collapse). EMA fixes this by maintaining a separate
   slowly-moving copy of the encoder as the target. The target updates
   via Exponential Moving Average — it changes very slowly, providing
   a stable prediction target that the student encoder cannot cheat against.

   This is how V-JEPA 2.1 was actually trained by Meta.

2. MOTION-WEIGHTED MASKING (forces learning physical dynamics)
   In v1/v2, frames were masked randomly. Lab videos have mostly static
   backgrounds (camera fixed on bench), so the model could predict masked
   frames by copying the static background — learning nothing about the
   actual procedure. Motion-weighted masking preferentially masks frames
   with high motion (hands moving, liquids transferring, equipment being
   adjusted), forcing the model to learn physical dynamics.

Configuration:
   Set USE_EMA and USE_MOTION_MASKING at the top to run different
   combinations. Each run takes ~5 minutes on H100.

   Run 1: USE_EMA=True,  USE_MOTION_MASKING=False  (EMA only)
   Run 2: USE_EMA=False, USE_MOTION_MASKING=True   (Motion masking only)
   Run 3: USE_EMA=True,  USE_MOTION_MASKING=True   (Both — strongest)
"""

import os
import sys
import json
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, '/home/.cache/torch/hub/facebookresearch_vjepa2_main')
sys.path.insert(0, '/home/tacit_experiment')

from dataset import LabProcDataset

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGURATION — change these to run different combinations
# ══════════════════════════════════════════════════════════════════════════════
USE_EMA            = True     # True = EMA target encoder (prevents collapse)
USE_MOTION_MASKING = True     # True = mask moving parts preferentially
EMA_MOMENTUM       = 0.996   # how slowly the target encoder updates (0.996 = very slow)
MOTION_FLOOR       = 0.1     # minimum masking probability for static frames
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH = Path("/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE      = 4
N_EPOCHS        = 7        # 7 epochs — more data (1011 videos) supports longer training; EMA prevents collapse
LR              = 5e-6
BLOCKS_TO_TRAIN = 3
MASK_RATIO      = 0.5
NUM_WORKERS     = 4

# Auto-name the experiment based on configuration
EXPERIMENT_NAME = "v3"
if USE_EMA and USE_MOTION_MASKING:
    EXPERIMENT_NAME += "_ema_motion"
elif USE_EMA:
    EXPERIMENT_NAME += "_ema_only"
elif USE_MOTION_MASKING:
    EXPERIMENT_NAME += "_motion_only"
else:
    EXPERIMENT_NAME += "_baseline"

OUTPUT_DIR = Path(f"/home/tacit_experiment/adapted_{EXPERIMENT_NAME}_checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Model Building ────────────────────────────────────────────────────────────

def build_encoder():
    from app.vjepa_2_1.models import vision_transformer as vit_encoder
    encoder = vit_encoder.__dict__['vit_large'](
        patch_size=16,
        img_size=(384, 384),
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    return encoder


def load_pretrained_weights(encoder):
    print(f"[Adapt] Loading checkpoint from {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    encoder_state = ckpt['ema_encoder']
    clean_state = {k.replace('module.', '').replace('backbone.', ''): v
                   for k, v in encoder_state.items()}
    missing, unexpected = encoder.load_state_dict(clean_state, strict=False)
    print(f"[Adapt] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    return encoder


def freeze_except_last_n(encoder, n: int):
    for p in encoder.parameters():
        p.requires_grad = False

    blocks = None
    if hasattr(encoder, 'blocks'):
        blocks = list(encoder.blocks)
    elif hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'blocks'):
        blocks = list(encoder.encoder.blocks)
    if blocks is None:
        for name, module in encoder.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 10:
                blocks = list(module)
                break
    if blocks is None:
        raise RuntimeError("Cannot find transformer blocks.")

    print(f"[Adapt] Total blocks: {len(blocks)}, unfreezing last {n}")
    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad = True
    for name, p in encoder.named_parameters():
        if 'norm' in name.lower() and 'block' not in name.lower():
            p.requires_grad = True

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    print(f"[Adapt] Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
          f"({trainable/total*100:.1f}%)")
    return encoder


# ── EMA Target Encoder ────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(student, teacher, momentum):
    """
    Update the teacher (target) encoder using Exponential Moving Average.

    teacher_params = momentum * teacher_params + (1 - momentum) * student_params

    With momentum=0.996, the teacher changes very slowly — only 0.4% of its
    weights come from the student at each step. This provides a stable
    prediction target that prevents representation collapse.

    This is exactly how V-JEPA 2.1 was trained by Meta.
    """
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)


# ── Motion-Weighted Masking ───────────────────────────────────────────────────

def compute_motion_weights(frames, floor=MOTION_FLOOR):
    """
    Compute per-frame motion magnitude using frame differencing.
    Frames with more motion get higher weight = more likely to be masked.

    This forces the model to predict the MOVING parts (hands, liquids,
    equipment being manipulated) rather than coasting on the static
    background (bench surface, wall, equipment not in use).

    frames: (B, 3, T, H, W) tensor
    Returns: (B, T) tensor of weights

    How it works:
    1. Compute L2 difference between consecutive frames
    2. Frames with large differences (= lots of motion) get high weight
    3. Add a floor so static frames still have some chance of being masked
    4. Normalise to probabilities
    """
    # frames is (B, C, T, H, W)
    B, C, T, H, W = frames.shape

    # Compute difference between consecutive frames
    # frames[:, :, 1:] = frames at time 1,2,...,T-1
    # frames[:, :, :-1] = frames at time 0,1,...,T-2
    diffs = (frames[:, :, 1:] - frames[:, :, :-1]).pow(2).mean(dim=(1, 3, 4))
    # diffs shape: (B, T-1) — motion magnitude per frame transition

    # First frame has no predecessor — assign zero motion
    zeros = torch.zeros(B, 1, device=frames.device, dtype=diffs.dtype)
    motion = torch.cat([zeros, diffs], dim=1)  # (B, T)

    # Add floor so static frames still have some masking probability
    # Without this, the model would never see predictions for static regions
    weights = motion + floor

    # Normalise to probabilities per sample
    weights = weights / weights.sum(dim=1, keepdim=True)

    return weights


def create_motion_mask(frames, n_mask_frames, floor=MOTION_FLOOR):
    """
    Create a binary mask where frames with more motion are more likely
    to be masked.

    frames: (B, 3, T, H, W)
    n_mask_frames: how many frames to mask per sample
    Returns: (B, T) boolean tensor — True = masked
    """
    B, C, T, H, W = frames.shape
    weights = compute_motion_weights(frames, floor)

    mask = torch.zeros(B, T, device=frames.device, dtype=torch.bool)
    for i in range(B):
        # Sample frames to mask, weighted by motion magnitude
        # replacement=False ensures each frame masked at most once
        chosen = torch.multinomial(weights[i], min(n_mask_frames, T),
                                   replacement=False)
        mask[i, chosen] = True

    return mask


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(student_encoder, target_encoder, loader, optimizer, scaler, epoch):
    student_encoder.train()
    if target_encoder is not None:
        target_encoder.eval()

    total_loss = 0
    n_batches = 0
    t0 = time.time()

    for batch_idx, (frames, labels, paths) in enumerate(loader):
        frames = frames.to(DEVICE)
        B, C, T, H, W = frames.shape

        # ── Create mask ───────────────────────────────────────────────
        n_mask = int(T * MASK_RATIO)

        if USE_MOTION_MASKING:
            mask = create_motion_mask(frames, n_mask)
        else:
            # Random masking (v1/v2 behavior)
            mask = torch.rand(B, T, device=DEVICE) < MASK_RATIO

        # Ensure at least 1 masked and 1 visible per sample
        for i in range(B):
            if mask[i].all():
                mask[i, 0] = False
            if not mask[i].any():
                mask[i, -1] = True

        # Zero out masked frames in context input
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        context_frames = frames * (~mask_expanded).float()

        # ── Forward pass ──────────────────────────────────────────────
        with torch.amp.autocast('cuda'):
            # Get TARGET representations (what we want to predict)
            with torch.no_grad():
                if USE_EMA and target_encoder is not None:
                    # EMA target — stable, prevents collapse
                    full_repr = target_encoder(frames)
                else:
                    # Same encoder as target (v1/v2 — prone to collapse)
                    full_repr = student_encoder(frames)
                if isinstance(full_repr, (list, tuple)):
                    full_repr = full_repr[0]

            # Get STUDENT representations (from partial/context input)
            ctx_repr = student_encoder(context_frames)
            if isinstance(ctx_repr, (list, tuple)):
                ctx_repr = ctx_repr[0]

            # ── Build token-level mask ────────────────────────────────
            n_temporal = T // 2  # tubelet_size = 2
            n_spatial = (H // 16) * (W // 16)  # 24 * 24 = 576

            tubelet_mask = torch.zeros(B, n_temporal, device=DEVICE, dtype=torch.bool)
            for t in range(n_temporal):
                f0 = t * 2
                f1 = min(t * 2 + 1, T - 1)
                tubelet_mask[:, t] = mask[:, f0] | mask[:, f1]

            token_mask = tubelet_mask.unsqueeze(-1).expand(-1, -1, n_spatial)
            token_mask = token_mask.reshape(B, -1)  # (B, 4608)

            # ── Compute loss on masked tokens only ────────────────────
            masked_full = full_repr[token_mask]
            masked_ctx = ctx_repr[token_mask]

            if masked_full.numel() == 0:
                continue

            loss = nn.functional.mse_loss(masked_ctx, masked_full.detach())

        # ── Backward pass ─────────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in student_encoder.parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()

        # ── Update EMA target ─────────────────────────────────────────
        if USE_EMA and target_encoder is not None:
            update_ema(student_encoder, target_encoder, EMA_MOMENTUM)

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 20 == 0:
            elapsed = time.time() - t0
            vram_gb = torch.cuda.memory_allocated() / 1e9
            mask_type = "motion" if USE_MOTION_MASKING else "random"
            target_type = "EMA" if USE_EMA else "self"
            print(f"  E{epoch} B{batch_idx:>4d}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"VRAM: {vram_gb:.1f}GB | "
                  f"mask={mask_type} target={target_type} | "
                  f"{elapsed:.0f}s")

    return total_loss / max(n_batches, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"TACIT — V-JEPA 2.1 Domain Adaptation ({EXPERIMENT_NAME})")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    print(f"Configuration:")
    print(f"  USE_EMA:            {USE_EMA}")
    print(f"  USE_MOTION_MASKING: {USE_MOTION_MASKING}")
    print(f"  EMA_MOMENTUM:       {EMA_MOMENTUM}")
    print(f"  Epochs:             {N_EPOCHS}")
    print(f"  Learning rate:      {LR}")
    print(f"  Unfrozen blocks:    {BLOCKS_TO_TRAIN}")
    print(f"  Experiment name:    {EXPERIMENT_NAME}")
    print(f"  Output dir:         {OUTPUT_DIR}")
    print()

    # ── Build student encoder ─────────────────────────────────────────────
    print("[Adapt] Building student encoder...")
    student = build_encoder()
    student = load_pretrained_weights(student)
    student = freeze_except_last_n(student, BLOCKS_TO_TRAIN)
    student = student.to(DEVICE)

    # ── Build EMA target encoder (if enabled) ─────────────────────────────
    target = None
    if USE_EMA:
        print("[Adapt] Building EMA target encoder (deep copy of student)...")
        target = copy.deepcopy(student)
        # Target encoder never receives gradients
        for p in target.parameters():
            p.requires_grad = False
        target = target.to(DEVICE)
        print(f"[Adapt] EMA momentum: {EMA_MOMENTUM}")
        print(f"[Adapt] Target encoder: frozen, updated via EMA after each step")
    else:
        print("[Adapt] No EMA — using same encoder as target (v1/v2 style)")

    print(f"[Adapt] VRAM after model load: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print()

    # ── Load datasets ─────────────────────────────────────────────────────
    print("[Adapt] Loading datasets...")
    train_ds = LabProcDataset(split="train")

    if len(train_ds) == 0:
        print("[Adapt] ERROR: No training videos found.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
        drop_last=True,
    )

    print(f"[Adapt] Training batches per epoch: {len(train_loader)}")
    print()

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    # ── Training loop ─────────────────────────────────────────────────────
    history = []
    best_loss = float('inf')

    for epoch in range(1, N_EPOCHS + 1):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch}/{N_EPOCHS}  |  LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'=' * 60}")

        t_start = time.time()
        train_loss = train_epoch(student, target, train_loader, optimizer,
                                 scaler, epoch)
        t_elapsed = time.time() - t_start

        scheduler.step()

        print(f"\n[Adapt] Epoch {epoch} complete | "
              f"Loss: {train_loss:.4f} | Time: {t_elapsed/60:.1f} min")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": scheduler.get_last_lr()[0],
            "time_min": round(t_elapsed / 60, 1),
        })

        # Save checkpoint
        ckpt_path = OUTPUT_DIR / f"adapted_{EXPERIMENT_NAME}_epoch{epoch:02d}.pth"
        torch.save({
            "epoch": epoch,
            "model_state": student.state_dict(),
            "train_loss": train_loss,
            "config": {
                "use_ema": USE_EMA,
                "use_motion_masking": USE_MOTION_MASKING,
                "ema_momentum": EMA_MOMENTUM,
                "lr": LR,
                "blocks_to_train": BLOCKS_TO_TRAIN,
                "n_epochs": N_EPOCHS,
            }
        }, ckpt_path)
        print(f"[Adapt] Saved: {ckpt_path}")

        if train_loss < best_loss:
            best_loss = train_loss
            best_path = OUTPUT_DIR / f"adapted_{EXPERIMENT_NAME}_best.pth"
            torch.save(student.state_dict(), best_path)
            print(f"[Adapt] New best! Loss: {best_loss:.4f}")

        with open(OUTPUT_DIR / f"history_{EXPERIMENT_NAME}.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE — {EXPERIMENT_NAME}")
    print(f"{'=' * 60}")
    print(f"USE_EMA:            {USE_EMA}")
    print(f"USE_MOTION_MASKING: {USE_MOTION_MASKING}")
    print(f"Best loss:          {best_loss:.4f}")
    print(f"Best checkpoint:    {OUTPUT_DIR}/adapted_{EXPERIMENT_NAME}_best.pth")
    total_time = sum(h['time_min'] for h in history)
    print(f"Total time:         {total_time:.0f} min ({total_time/60:.1f} hrs)")


if __name__ == "__main__":
    main()