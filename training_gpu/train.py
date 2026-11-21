"""
FitnessVision GPU Pose Estimation Trainer
- GPU (CUDA) with Mixed Precision (AMP FP16)
- MobileNetV2 with Discriminative Learning Rates + Cosine Annealing
- Anti-stuck watchdog: moves to next epoch if batch exceeds timeout
- 6-hour total training time hard limit with safe saving
- PCK@3 evaluation + auto-save best model
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training_gpu.models import FitnessVisionPoseNet, MaskedHeatmapMSELoss
from training_gpu.datasets import create_dataloaders

# Configuration
DATASET_DIR = PROJECT_ROOT / "datasets" / "MPII Dataset" / "Full_MPII_Dataset" / "full_16_joint_dataset"
OUTPUT_DIR = PROJECT_ROOT / "training_outputs" / "gpu_v3"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"
REPORT_PATH = OUTPUT_DIR / "training_report.txt"
BEST_MODEL_PATH = OUTPUT_DIR / "best_pose_model.pth"
LATEST_MODEL_PATH = OUTPUT_DIR / "latest_pose_model.pth"

BATCH_SIZE = 16
NUM_EPOCHS = 100
MAX_TRAINING_SECONDS = 6.0 * 3600  # 6 hours
BATCH_TIMEOUT_SECONDS = 90.0  # Max seconds per batch before flagging stuck
BACKBONE_LR = 1e-4
DECODER_LR = 5e-4
WEIGHT_DECAY = 1e-4
PCK_THRESHOLD = 3.0  # pixels on 64x64 heatmap


def calculate_pck(preds, targets, masks, threshold=PCK_THRESHOLD):
    """
    Computes PCK across batch on 64x64 heatmaps.
    preds: [B, 16, 64, 64]
    targets: [B, 16, 64, 64]
    masks: [B, 16]
    """
    correct = 0
    total = 0

    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()

    batch_size, num_joints = masks_np.shape

    for b in range(batch_size):
        for j in range(num_joints):
            if masks_np[b, j] <= 0.5:
                continue

            pred_hm = preds_np[b, j]
            target_hm = targets_np[b, j]

            py, px = np.unravel_index(np.argmax(pred_hm), pred_hm.shape)
            ty, tx = np.unravel_index(np.argmax(target_hm), target_hm.shape)

            dist = np.sqrt((px - tx) ** 2 + (py - ty) ** 2)
            if dist <= threshold:
                correct += 1
            total += 1

    return correct, total


def evaluate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_batches = 0
    total_correct = 0
    total_joints = 0

    with torch.no_grad():
        for images, heatmaps, masks in val_loader:
            images = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast():
                preds = model(images)
                loss = criterion(preds, heatmaps, masks)

            val_loss += loss.item()
            val_batches += 1

            correct, joints = calculate_pck(preds, heatmaps, masks)
            total_correct += correct
            total_joints += joints

    mean_loss = val_loss / max(val_batches, 1)
    pck = (total_correct / total_joints * 100.0) if total_joints > 0 else 0.0
    return mean_loss, pck, total_joints


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FitnessVision GPU Training - RTX 2050 at Maximum Acceleration")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True  # Auto-tuner for max GPU performance

    print(f"Dataset: {DATASET_DIR}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")

    # Build DataLoaders
    print("Loading datasets...")
    train_loader, val_loader, n_train, n_val = create_dataloaders(
        DATASET_DIR, batch_size=BATCH_SIZE, val_split=0.2
    )
    print(f"Training samples: {n_train:,} ({len(train_loader)} batches)")
    print(f"Validation samples: {n_val:,} ({len(val_loader)} batches)")

    # Initialize Model
    print("Building model (MobileNetV2 backbone + FPN Decoder)...")
    model = FitnessVisionPoseNet(num_joints=16, pretrained=True).to(device)

    # Optimizer with discriminative learning rates
    optimizer = torch.optim.AdamW([
        {"params": model.stage1.parameters(), "lr": BACKBONE_LR * 0.5},
        {"params": model.stage2.parameters(), "lr": BACKBONE_LR * 0.5},
        {"params": model.stage3.parameters(), "lr": BACKBONE_LR},
        {"params": model.stage4.parameters(), "lr": BACKBONE_LR},
        {"params": model.dec_bottleneck.parameters(), "lr": DECODER_LR},
        {"params": model.dec1.parameters(), "lr": DECODER_LR},
        {"params": model.dec2.parameters(), "lr": DECODER_LR},
        {"params": model.dec3.parameters(), "lr": DECODER_LR},
        {"params": model.final_conv.parameters(), "lr": DECODER_LR},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine Annealing LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    criterion = MaskedHeatmapMSELoss()
    scaler = GradScaler()  # For FP16 mixed precision

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pck": [],
        "lr": [],
        "epoch_seconds": [],
        "stuck_warnings": []
    }

    start_training_time = time.time()
    best_pck = 0.0
    best_val_loss = float("inf")

    print("\nStarting training loop...")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        # Check 6-hour limit
        elapsed_total = time.time() - start_training_time
        if elapsed_total >= MAX_TRAINING_SECONDS:
            print(f"\n[TIME LIMIT REACHED] Training reached 6 hours ({elapsed_total/3600:.2f} hrs). Stopping safely.")
            break

        model.train()
        train_loss = 0.0
        train_batches = 0
        epoch_stuck = False

        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} [Elapsed: {elapsed_total/60:.1f}m] ---")

        for batch_idx, (images, heatmaps, masks) in enumerate(train_loader):
            batch_start = time.time()

            # Watchdog check
            images = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # AMP forward pass
            with autocast():
                preds = model(images)
                loss = criterion(preds, heatmaps, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_batches += 1

            batch_elapsed = time.time() - batch_start
            if batch_elapsed > BATCH_TIMEOUT_SECONDS:
                msg = f"Epoch {epoch} Batch {batch_idx} exceeded timeout ({batch_elapsed:.1f}s > {BATCH_TIMEOUT_SECONDS}s). Skipping to next epoch."
                print(f"\n[STUCK WARNING] {msg}")
                history["stuck_warnings"].append(msg)
                epoch_stuck = True
                break

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                avg_running = train_loss / train_batches
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {avg_running:.6f} ({batch_elapsed*1000:.1f}ms/batch)")

        scheduler.step()
        epoch_duration = time.time() - epoch_start_time

        # Validation
        val_loss, val_pck, evaluated_joints = evaluate(model, criterion, val_loader, device)

        mean_train_loss = train_loss / max(train_batches, 1)
        current_lr = optimizer.param_groups[-1]["lr"]

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(val_loss)
        history["val_pck"].append(val_pck)
        history["lr"].append(current_lr)
        history["epoch_seconds"].append(epoch_duration)

        print(f"Epoch {epoch} Summary: Train Loss: {mean_train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PCK@3: {val_pck:.2f}% | Time: {epoch_duration:.1f}s")

        # Save Latest Checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_pck": val_pck
        }, LATEST_MODEL_PATH)

        # Save Best Model
        if val_pck > best_pck or (abs(val_pck - best_pck) < 0.1 and val_loss < best_val_loss):
            best_pck = val_pck
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_pck": val_pck
            }, BEST_MODEL_PATH)
            print(f"  >>> [NEW BEST MODEL] Saved with PCK: {best_pck:.2f}% at epoch {epoch}")

        # Save periodic checkpoint
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            ckpt_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "pck": val_pck}, ckpt_path)

        # Persist history
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    total_training_minutes = (time.time() - start_training_time) / 60.0

    # Final Report
    report_lines = [
        "FitnessVision GPU Training V3 Report",
        "=" * 45,
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"CUDA Version: {torch.version.cuda}",
        f"Total Samples: {n_train + n_val:,} (Train: {n_train:,}, Val: {n_val:,})",
        f"Epochs Completed: {len(history['train_loss'])}/{NUM_EPOCHS}",
        f"Best Validation PCK@3: {best_pck:.2f}%",
        f"Best Validation Loss: {best_val_loss:.6f}",
        f"Total Training Duration: {total_training_minutes:.2f} minutes",
        f"Best Model File: {BEST_MODEL_PATH}",
        f"Latest Model File: {LATEST_MODEL_PATH}",
        f"History File: {HISTORY_PATH}",
        f"Stuck Incidents: {len(history['stuck_warnings'])}",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()
