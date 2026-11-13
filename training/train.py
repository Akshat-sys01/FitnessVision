import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf

from data_loader import train_dataset, val_dataset, NUM_JOINTS
from model import build_pose_model
from loss import masked_mse

EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 8
OUTPUT_DIR = Path("training_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = OUTPUT_DIR / "best_pose_model.keras"
FINAL_MODEL_PATH = OUTPUT_DIR / "final_pose_model.keras"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"

model = build_pose_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def train_step(images, true_heatmaps, masks):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = masked_mse(true_heatmaps, predictions, masks)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, tf.reduce_sum(masks)

@tf.function
def validation_step(images, true_heatmaps, masks):
    predictions = model(images, training=False)
    loss = masked_mse(true_heatmaps, predictions, masks)
    joint_losses = tf.reduce_mean(
        tf.square(true_heatmaps - predictions), axis=[1, 2]
    )
    return loss, joint_losses

history = {"train_loss": [], "val_loss": [], "joint_val_loss": []}
best_val_loss = float("inf")
epochs_without_improvement = 0

print("Starting FitnessVision training...")
model.summary()

for epoch in range(EPOCHS):
    start = time.time()

    train_sum = 0.0
    train_visible = 0.0
    for images, heatmaps, masks in train_dataset:
        masks = tf.cast(masks, tf.float32)
        loss_value, visible = train_step(images, heatmaps, masks)
        count = float(visible.numpy())
        train_sum += float(loss_value.numpy()) * count
        train_visible += count
    train_loss = train_sum / train_visible if train_visible else 0.0

    val_sum = 0.0
    val_visible = 0.0
    joint_sums = np.zeros(NUM_JOINTS, dtype=np.float64)
    joint_counts = np.zeros(NUM_JOINTS, dtype=np.float64)

    for images, heatmaps, masks in val_dataset:
        masks = tf.cast(masks, tf.float32)
        loss_value, joint_losses = validation_step(images, heatmaps, masks)
        masks_np = masks.numpy()
        joint_np = joint_losses.numpy()
        count = float(masks_np.sum())
        val_sum += float(loss_value.numpy()) * count
        val_visible += count
        joint_sums += (joint_np * masks_np).sum(axis=0)
        joint_counts += masks_np.sum(axis=0)

    val_loss = val_sum / val_visible if val_visible else float("inf")
    joint_avg = np.divide(
        joint_sums, joint_counts,
        out=np.zeros(NUM_JOINTS), where=joint_counts > 0
    )

    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))
    history["joint_val_loss"].append(joint_avg.tolist())

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"train={train_loss:.6f} | val={val_loss:.6f} | "
        f"time={time.time() - start:.1f}s"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        model.save(BEST_MODEL_PATH)
        print("  Best model saved.")
    else:
        epochs_without_improvement += 1
        print(f"  No improvement: {epochs_without_improvement}/{PATIENCE}")

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break

model.save(FINAL_MODEL_PATH)
with open(HISTORY_PATH, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=4)

print("\nTraining completed.")
print("Best validation loss:", best_val_loss)
print("Best model:", BEST_MODEL_PATH.resolve())
print("Final model:", FINAL_MODEL_PATH.resolve())
print("History:", HISTORY_PATH.resolve())
