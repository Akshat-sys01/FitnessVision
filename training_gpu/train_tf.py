"""
FitnessVision TensorFlow Pose Estimation Training Pipeline
===========================================================
This module provides the complete training, validation, watchdog, and checkpointing
engine built exclusively on TensorFlow 2 / Keras, replacing the PyTorch trainer.

Key Technical Mechanisms:
-------------------------
1. Custom Training Step with tf.GradientTape:
   - Evaluates the forward pass, calculates masked MSE loss, and extracts exact gradients.
   - Applies global norm gradient clipping (5.0) to stabilize backpropagation through deep skips.
   - Updates weights using tf.keras.optimizers.AdamW with decoupled weight decay.
2. Cosine Decay Learning Rate Scheduling:
   - Uses tf.keras.optimizers.schedules.CosineDecay to smoothly lower learning rate from
     5e-4 to 1e-6 over 100 epochs, allowing rapid global convergence followed by precision settling.
3. PCK@3 Evaluation Metric (Percentage of Correct Keypoints):
   - Computes argmax coordinates (x, y) on predicted vs ground-truth 64x64 heatmaps.
   - Validates whether euclidean pixel error is <= 3.0 heatmap pixels on all visible joints.
4. Anti-Stuck Watchdog & 6-Hour Timeout Protection:
   - Tracks time elapsed per batch and epoch. If any batch hangs beyond timeout threshold,
     the loop reports a warning and safely skips to the next epoch.
   - Monitors cumulative training duration and cleanly saves best & latest models before 6h limit.
5. Persistent Artifacts & Diagnostics:
   - Saves best model in Keras 3 format (.keras) and weights checkpoint.
   - Produces JSON training history and comprehensive text summary report.
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf

# Suppress verbose C++ engine warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Configure root directory in Python search path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training_gpu.models_tf import build_pose_model, MaskedHeatmapMSELoss
from training_gpu.datasets_tf import load_dataset_records, create_tf_dataset

# ================= Configuration Parameters =================
DATASET_DIR = PROJECT_ROOT / "datasets" / "MPII Dataset" / "Full_MPII_Dataset" / "full_16_joint_dataset"
OUTPUT_DIR = PROJECT_ROOT / "training_outputs" / "gpu_v3_tf"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
HISTORY_PATH = OUTPUT_DIR / "training_history_tf.json"
REPORT_PATH = OUTPUT_DIR / "training_report_tf.txt"
BEST_MODEL_PATH = OUTPUT_DIR / "best_pose_model_tf.keras"
LATEST_MODEL_PATH = OUTPUT_DIR / "latest_pose_model_tf.keras"

BATCH_SIZE = 16
NUM_EPOCHS = 100
MAX_TRAINING_SECONDS = 6.0 * 3600  # Strict 6-hour ceiling
BATCH_TIMEOUT_SECONDS = 90.0       # Stuck detection per batch
INITIAL_LR = 5e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
PCK_THRESHOLD = 3.0  # Measured in 64x64 heatmap pixels


def compute_batch_pck(
    predictions: np.ndarray,
    packed_targets: np.ndarray,
    threshold: float = PCK_THRESHOLD
) -> Tuple[int, int]:
    """
    Computes Percentage of Correct Keypoints (PCK) across a single batch.

    Technical calculation:
        1. Slices heatmaps (first 16 channels) and masks (last 16 channels).
        2. Computes argmax across 2D spatial dimensions to find peak joint coordinates.
        3. Calculates Euclidean distance: sqrt((pred_x - true_x)^2 + (pred_y - true_y)^2).
        4. Compares distance against the 3.0-pixel threshold for all visible keypoints.

    Returns:
        (correct_count, total_evaluated_count)
    """
    batch_size = predictions.shape[0]
    num_joints = 16

    true_heatmaps = packed_targets[..., :num_joints]
    masks = packed_targets[:, 0, 0, num_joints:]  # Joint visibility mask (batch, 16)

    correct = 0
    total = 0

    for b in range(batch_size):
        for j in range(num_joints):
            if masks[b, j] <= 0.5:
                # Keypoint is occluded or outside bounding box; skip evaluation
                continue

            pred_hm = predictions[b, :, :, j]
            true_hm = true_heatmaps[b, :, :, j]

            py, px = np.unravel_index(np.argmax(pred_hm), pred_hm.shape)
            ty, tx = np.unravel_index(np.argmax(true_hm), true_hm.shape)

            dist = np.hypot(float(px - tx), float(py - ty))
            if dist <= threshold:
                correct += 1
            total += 1

    return correct, total


def evaluate_dataset(
    model: tf.keras.Model,
    loss_fn: tf.keras.losses.Loss,
    val_dataset: tf.data.Dataset
) -> Tuple[float, float, int]:
    """
    Runs full validation pass over val_dataset in inference mode.

    Returns:
        (mean_loss, pck_percentage, total_joints_evaluated)
    """
    val_loss_sum = 0.0
    val_batches = 0
    total_correct = 0
    total_joints = 0

    for x_batch, y_batch in val_dataset:
        # Forward pass without gradient tracking
        preds = model(x_batch, training=False)
        loss_val = loss_fn(y_batch, preds)

        val_loss_sum += float(loss_val.numpy())
        val_batches += 1

        correct, joints = compute_batch_pck(preds.numpy(), y_batch.numpy())
        total_correct += correct
        total_joints += joints

    mean_loss = val_loss_sum / max(val_batches, 1)
    pck = (total_correct / total_joints * 100.0) if total_joints > 0 else 0.0
    return mean_loss, pck, total_joints


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("FitnessVision TensorFlow Training Pipeline (Keras 3 Engine)")
    print("=" * 65)

    physical_gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow Version:", tf.__version__)
    print("Available Physical Devices:", tf.config.list_physical_devices())

    # Build Data Pipelines
    print(f"\nLoading dataset records from: {DATASET_DIR}")
    train_records, val_records = load_dataset_records(DATASET_DIR, val_split=0.2)
    n_train = len(train_records)
    n_val = len(val_records)
    num_train_batches = n_train // BATCH_SIZE

    print(f"Training samples: {n_train:,} (~{num_train_batches} batches at batch size {BATCH_SIZE})")
    print(f"Validation samples: {n_val:,}")

    train_ds = create_tf_dataset(train_records, DATASET_DIR, batch_size=BATCH_SIZE, is_training=True)
    val_ds = create_tf_dataset(val_records, DATASET_DIR, batch_size=BATCH_SIZE, is_training=False)

    # Instantiate Model
    print("\nConstructing ResNet-50 + Multi-scale FPN Decoder in TensorFlow...")
    model = build_pose_model(input_shape=(256, 256, 3), num_joints=16, trainable_backbone=True)
    model.summary()

    # Learning Rate Schedule & Optimizer
    total_steps = num_train_batches * NUM_EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INITIAL_LR,
        decay_steps=total_steps,
        alpha=MIN_LR / INITIAL_LR
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=5.0  # Global gradient norm clipping
    )

    loss_fn = MaskedHeatmapMSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pck": [],
        "learning_rate": [],
        "epoch_seconds": [],
        "stuck_warnings": []
    }

    start_training_time = time.time()
    best_pck = 0.0
    best_val_loss = float("inf")

    # TensorFlow Graph-Compiled Training Step for performance acceleration
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss_value = loss_fn(y, predictions)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss_value

    print("\nStarting TensorFlow training loop...")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        elapsed_total = time.time() - start_training_time

        # Check 6-Hour Ceiling
        if elapsed_total >= MAX_TRAINING_SECONDS:
            print(f"\n[TIME LIMIT REACHED] Elapsed {elapsed_total/3600:.2f} hours. Halting training safely.")
            break

        train_loss_sum = 0.0
        batches_processed = 0
        epoch_aborted = False

        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} [Elapsed: {elapsed_total/60:.1f}m] ---")

        for batch_idx, (x_batch, y_batch) in enumerate(train_ds):
            batch_start = time.time()

            # Execute compiled training step
            loss_val = train_step(x_batch, y_batch)
            train_loss_sum += float(loss_val.numpy())
            batches_processed += 1

            batch_elapsed = time.time() - batch_start

            # Watchdog: Stuck batch protection
            if batch_elapsed > BATCH_TIMEOUT_SECONDS:
                msg = f"Epoch {epoch} Batch {batch_idx+1} took {batch_elapsed:.1f}s (> {BATCH_TIMEOUT_SECONDS}s). Skipping to next epoch."
                print(f"\n[STUCK WARNING] {msg}")
                history["stuck_warnings"].append(msg)
                epoch_aborted = True
                break

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_train_batches:
                running_loss = train_loss_sum / batches_processed
                print(f"  Batch {batch_idx+1}/{num_train_batches} - Loss: {running_loss:.6f} ({batch_elapsed*1000:.1f}ms/batch)")

        epoch_duration = time.time() - epoch_start_time

        # Run Validation Evaluation
        print("  Evaluating validation set...")
        val_loss, val_pck, val_joints = evaluate_dataset(model, loss_fn, val_ds)

        mean_train_loss = train_loss_sum / max(batches_processed, 1)
        current_lr = float(optimizer.learning_rate(optimizer.iterations).numpy()) if callable(optimizer.learning_rate) else float(optimizer.learning_rate.numpy())

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(val_loss)
        history["val_pck"].append(val_pck)
        history["learning_rate"].append(current_lr)
        history["epoch_seconds"].append(epoch_duration)

        print(f"Epoch {epoch} Summary: Train Loss: {mean_train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PCK@3: {val_pck:.2f}% | Time: {epoch_duration:.1f}s")

        # Save Latest Model
        model.save(LATEST_MODEL_PATH)

        # Save Best Model Checkpoint
        if val_pck > best_pck or (abs(val_pck - best_pck) < 0.1 and val_loss < best_val_loss):
            best_pck = val_pck
            best_val_loss = val_loss
            model.save(BEST_MODEL_PATH)
            print(f"  >>> [NEW BEST MODEL] Saved to {BEST_MODEL_PATH.name} with Val PCK: {best_pck:.2f}%")

        # Save periodic weights checkpoint
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            ckpt_file = CHECKPOINT_DIR / f"ckpt_epoch_{epoch:03d}.weights.h5"
            model.save_weights(ckpt_file)

        # Update JSON history
        with HISTORY_PATH.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    total_training_minutes = (time.time() - start_training_time) / 60.0

    # Write Final Training Summary Report
    report_lines = [
        "FitnessVision TensorFlow (Keras 3) Training Report",
        "=" * 50,
        f"Framework: TensorFlow {tf.__version__} / Keras",
        f"Total Samples: {n_train + n_val:,} (Train: {n_train:,}, Val: {n_val:,})",
        f"Epochs Completed: {len(history['train_loss'])}/{NUM_EPOCHS}",
        f"Best Validation PCK@3: {best_pck:.2f}%",
        f"Best Validation Loss: {best_val_loss:.6f}",
        f"Total Duration: {total_training_minutes:.2f} minutes",
        f"Best Model File: {BEST_MODEL_PATH}",
        f"Latest Model File: {LATEST_MODEL_PATH}",
        f"History File: {HISTORY_PATH}",
        f"Stuck Incidents Detected: {len(history['stuck_warnings'])}",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    main()
