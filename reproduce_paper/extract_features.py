"""
extract_features.py — Extract Frozen V-JEPA 2.1 Features

Runs the encoder on all QC-passed videos and saves the output features
as .pt files. These features are then used for probe training.

Extracts features from TWO models:
  1. Base V-JEPA 2.1 (pretrained, no domain adaptation) = Condition B
  2. Domain-adapted V-JEPA 2.1 (after adapt.py) = Condition C

The feature files are much smaller than raw videos (~500MB vs 62GB),
making probe training fast and allowing transfer between machines.
"""

import torch
import sys
import time
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/.cache/torch/hub/facebookresearch_vjepa2_main')
sys.path.insert(0, '/home/tacit_experiment')

from dataset import LabProcDataset

DEVICE     = torch.device("cuda")
OUTPUT_DIR = Path("/home/tacit_features")
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_BASE    = Path("/home/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt")
CHECKPOINT_ADAPTED = Path("/home/tacit_experiment/adapted_checkpoints/adapted_best.pth")


def build_encoder():
    """Build V-JEPA 2.1 ViT-Large with exact pretrained config."""
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


def load_base_weights(encoder):
    """Load the original pretrained V-JEPA 2.1 weights (Condition B)."""
    ckpt = torch.load(CHECKPOINT_BASE, map_location='cpu')
    state = ckpt['ema_encoder']
    clean = {k.replace('module.', '').replace('backbone.', ''): v
             for k, v in state.items()}
    encoder.load_state_dict(clean, strict=False)
    print("[Extract] Base weights loaded")
    return encoder


def load_adapted_weights(encoder):
    """Load the domain-adapted weights (Condition C)."""
    state = torch.load(CHECKPOINT_ADAPTED, map_location='cpu')
    encoder.load_state_dict(state, strict=True)
    print("[Extract] Adapted weights loaded")
    return encoder


def extract_features(encoder, dataset, name: str, batch_size: int = 4):
    """
    Run encoder on all videos in dataset, mean-pool over tokens,
    save features and labels to disk.
    """
    encoder.eval()
    encoder = encoder.to(DEVICE)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_features = []
    all_labels   = []
    all_paths    = []

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch_idx, (frames, labels, paths) in enumerate(loader):
            # frames: (B, 3, T, 384, 384)
            frames = frames.to(DEVICE)
            out = encoder(frames)
            if isinstance(out, (list, tuple)):
                out = out[0]

            # Mean pool over spatiotemporal tokens -> (B, 1024)
            features = out.mean(dim=1).cpu().float()

            all_features.append(features)
            all_labels.extend(labels.tolist() if hasattr(labels, 'tolist') else labels)
            all_paths.extend(paths)

            if batch_idx % 20 == 0:
                print(f"  [{name}] Batch {batch_idx}/{len(loader)} "
                      f"({time.time()-t0:.0f}s)")

    all_features = torch.cat(all_features, dim=0)
    all_labels   = torch.tensor(all_labels)

    save_path = OUTPUT_DIR / f"features_{name}.pt"
    torch.save({
        "features": all_features,
        "labels":   all_labels,
        "paths":    all_paths,
    }, save_path)

    print(f"[Extract] Saved {name}: {all_features.shape} -> {save_path}")
    print(f"  File size: {save_path.stat().st_size / 1e6:.1f} MB")
    return all_features


def main():
    print("="*60)
    print("FEATURE EXTRACTION — Base vs Adapted V-JEPA 2.1")
    print("="*60)

    # Load datasets
    train_ds = LabProcDataset(split="train")
    val_ds   = LabProcDataset(split="val")

    # ── Condition B: Base V-JEPA 2.1 ──────────────────────────────────────────
    print("\n--- Extracting BASE features (Condition B) ---")
    encoder = build_encoder()
    encoder = load_base_weights(encoder)

    extract_features(encoder, train_ds, "base_train")
    extract_features(encoder, val_ds,   "base_val")

    del encoder
    torch.cuda.empty_cache()

    # ── Condition C: Domain-adapted V-JEPA 2.1 ───────────────────────────────
    if CHECKPOINT_ADAPTED.exists():
        print("\n--- Extracting ADAPTED features (Condition C) ---")
        encoder = build_encoder()
        encoder = load_adapted_weights(encoder)

        extract_features(encoder, train_ds, "adapted_train")
        extract_features(encoder, val_ds,   "adapted_val")

        del encoder
        torch.cuda.empty_cache()
    else:
        print(f"\n[Extract] Adapted checkpoint not found: {CHECKPOINT_ADAPTED}")
        print("[Extract] Run adapt.py first, then re-run this script.")

    print("\n[Extract] Done. Feature files saved to:", OUTPUT_DIR)
    for f in sorted(OUTPUT_DIR.glob("*.pt")):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint to extract features from")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Name for the output feature file (e.g. features_epoch3)")
    parser.add_argument("--checkpoint-key", type=str, default=None,
                        help="Key in checkpoint dict containing model state (e.g. model_state)")
    args = parser.parse_args()

    if args.checkpoint:
        # Single checkpoint mode — used by sweep_epochs.sh
        print(f"Extracting features from: {args.checkpoint}")
        encoder = build_encoder()

        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if args.checkpoint_key and isinstance(ckpt, dict) and args.checkpoint_key in ckpt:
            state = ckpt[args.checkpoint_key]
        elif isinstance(ckpt, dict) and 'model_state' in ckpt:
            state = ckpt['model_state']
        elif isinstance(ckpt, dict) and 'ema_encoder' in ckpt:
            state = ckpt['ema_encoder']
            state = {k.replace('module.', '').replace('backbone.', ''): v
                     for k, v in state.items()}
        else:
            state = ckpt

        encoder.load_state_dict(state, strict=False)

        train_ds = LabProcDataset(split="train")
        val_ds = LabProcDataset(split="val")

        # output_name should be the suffix AFTER "features_"
        # e.g. output_name="epoch1" → saves as features_epoch1_train.pt
        output_name = args.output_name or "custom"
        extract_features(encoder, train_ds, f"{output_name}_train")
        extract_features(encoder, val_ds, f"{output_name}_val")

        # Also save a combined file for train_probe.py compatibility
        # extract_features saves as features_{name}.pt, so the train file is:
        combined_path = OUTPUT_DIR / f"features_{output_name}.pt"
        train_data = torch.load(OUTPUT_DIR / f"features_{output_name}_train.pt")
        torch.save(train_data, combined_path)
        print(f"Combined: {combined_path}")
    else:
        # Default mode — extract both base and adapted
        main()