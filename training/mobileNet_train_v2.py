"""
FitnessVision Pose CNN V2
MobileNetV2 encoder + multiscale decoder.

Expected data_loader.py:
    train_dataset, val_dataset, NUM_JOINTS

Each batch must be (images, heatmaps, joint_masks).
Images: [B,256,256,3] normalized to [0,1]
Heatmaps: [B,64,64,16] (or [B,16,64,64])
Masks: [B,16], 1=visible, 0=ignored.

Run from Training:
    python train_v2.py

Resumable checkpoints are saved every 100 batches and every epoch.
"""

import json
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_loader import train_dataset, val_dataset, NUM_JOINTS


# ---------------- Configuration ----------------

INPUT_SIZE = 256
OUTPUT_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-5
CHECKPOINT_EVERY_STEPS = 100
EARLY_STOPPING_PATIENCE = 10
SEED = 42

OUT = Path("training_outputs_v2")
CKPT_DIR = OUT / "checkpoints"
BEST_MODEL = OUT / "best_pose_model_v2.keras"
FINAL_MODEL = OUT / "final_pose_model_v2.keras"
HISTORY_FILE = OUT / "training_history_v2.json"

keras.utils.set_random_seed(SEED)
OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


# ---------------- Model ----------------

def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_pose_model():
    image = keras.Input((INPUT_SIZE, INPUT_SIZE, 3), name="input_image")

    # data_loader supplies [0,1]; MobileNetV2 expects [-1,1].
    normalized = layers.Rescaling(2.0, offset=-1.0,
                                  name="mobilenet_preprocessing")(image)

    backbone = keras.applications.MobileNetV2(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False

    encoder = keras.Model(
        backbone.input,
        [
            backbone.get_layer("block_3_expand_relu").output,   # 64x64
            backbone.get_layer("block_6_expand_relu").output,   # 32x32
            backbone.get_layer("block_13_expand_relu").output,  # 16x16
            backbone.get_layer("out_relu").output,              # 8x8
        ],
        name="mobilenetv2_encoder"
    )

    s64, s32, s16, x = encoder(normalized)

    # Multiscale decoder with skip connections.
    x = conv_block(x, 256, "decoder_8")

    x = layers.UpSampling2D(2, interpolation="bilinear")(x)
    x = layers.Concatenate()([x, s16])
    x = conv_block(x, 192, "decoder_16")
    x = conv_block(x, 192, "decoder_16_refine")

    x = layers.UpSampling2D(2, interpolation="bilinear")(x)
    x = layers.Concatenate()([x, s32])
    x = conv_block(x, 128, "decoder_32")
    x = conv_block(x, 128, "decoder_32_refine")

    x = layers.UpSampling2D(2, interpolation="bilinear")(x)
    x = layers.Concatenate()([x, s64])
    x = conv_block(x, 96, "decoder_64")
    x = conv_block(x, 96, "decoder_64_refine")

    heatmaps = layers.Conv2D(
        NUM_JOINTS, 1, padding="same", activation=None,
        name="joint_heatmaps"
    )(x)

    return keras.Model(image, heatmaps, name="FitnessVision_Pose_V2"), backbone


# ---------------- Loss and data checks ----------------

def masked_mse(y_true, y_pred, masks):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    masks = tf.cast(masks, tf.float32)
    per_joint = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2])
    return tf.reduce_sum(per_joint * masks) / tf.maximum(tf.reduce_sum(masks), 1.0)


def unpack(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise ValueError("Dataset must yield (images, heatmaps, joint_masks).")
    images, heatmaps, masks = batch
    images = tf.cast(images, tf.float32)
    heatmaps = tf.cast(heatmaps, tf.float32)
    masks = tf.cast(masks, tf.float32)

    if heatmaps.shape.rank == 4 and heatmaps.shape[1] == NUM_JOINTS:
        heatmaps = tf.transpose(heatmaps, [0, 2, 3, 1])
    if masks.shape.rank == 3 and masks.shape[-1] == 1:
        masks = tf.squeeze(masks, axis=-1)
    return images, heatmaps, masks


def validate_dataset(dataset, name):
    images, heatmaps, masks = unpack(next(iter(dataset)))
    print(f"\n{name} batch:")
    print(" images:", images.shape, "range:",
          float(tf.reduce_min(images)), "to", float(tf.reduce_max(images)))
    print(" heatmaps:", heatmaps.shape)
    print(" masks:", masks.shape)

    tf.debugging.assert_equal(tf.shape(images)[1:], [256, 256, 3])
    tf.debugging.assert_equal(tf.shape(heatmaps)[1:], [64, 64, NUM_JOINTS])
    tf.debugging.assert_equal(tf.shape(masks)[1], NUM_JOINTS)

    lo, hi = float(tf.reduce_min(images)), float(tf.reduce_max(images))
    if lo < -0.01 or hi > 1.01:
        raise ValueError("Expected images normalized to [0,1]. Check data_loader.py.")


# ---------------- Build and resumable state ----------------

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())
print("Learning rate:", LEARNING_RATE)

model, backbone = build_pose_model()
print("Model output:", model.output_shape)
print("Total parameters:", f"{model.count_params():,}")

validate_dataset(train_dataset, "Training")
validate_dataset(val_dataset, "Validation")

optimizer = keras.optimizers.Adam(LEARNING_RATE)
epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False)
step_var = tf.Variable(0, dtype=tf.int64, trainable=False)
best_var = tf.Variable(float("inf"), dtype=tf.float32, trainable=False)
patience_var = tf.Variable(0, dtype=tf.int64, trainable=False)

checkpoint = tf.train.Checkpoint(
    model=model, optimizer=optimizer,
    epoch=epoch_var, step=step_var,
    best_val=best_var, patience=patience_var
)
manager = tf.train.CheckpointManager(
    checkpoint, str(CKPT_DIR), max_to_keep=3
)

history = {"train_loss": [], "val_loss": [], "learning_rate": [], "epoch_seconds": []}
if HISTORY_FILE.exists():
    try:
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        print("Existing history file unreadable; starting a new history log.")

if manager.latest_checkpoint:
    print("Restoring:", manager.latest_checkpoint)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    print("Resuming after epoch", int(epoch_var.numpy()),
          "| global step", int(step_var.numpy()))
else:
    print("No checkpoint found; starting fresh.")


# ---------------- Training functions ----------------

def train_batch(images, targets, masks):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = masked_mse(targets, predictions, masks)

    variables = model.trainable_variables
    grads = tape.gradient(loss, variables)
    pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
    grads, variables = zip(*pairs)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, variables))
    return loss


def evaluate(dataset):
    total, count = 0.0, 0
    for batch in dataset:
        images, targets, masks = unpack(batch)
        predictions = model(images, training=False)
        total += float(masked_mse(targets, predictions, masks).numpy())
        count += 1
    return total / max(count, 1)


def save_history():
    temp = HISTORY_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temp.replace(HISTORY_FILE)


# ---------------- Main training loop ----------------

for epoch_index in range(int(epoch_var.numpy()), EPOCHS):
    epoch_number = epoch_index + 1
    start = time.time()

    # The pretrained backbone is fine-tuned from the start with a conservative learning rate.

    total_loss, batches = 0.0, 0
    print(f"\n========== Epoch {epoch_number}/{EPOCHS} ==========")

    for batch in train_dataset:
        images, targets, masks = unpack(batch)
        loss = train_batch(images, targets, masks)
        total_loss += float(loss.numpy())
        batches += 1
        step_var.assign_add(1)

        if int(step_var.numpy()) % CHECKPOINT_EVERY_STEPS == 0:
            saved = manager.save(checkpoint_number=int(step_var.numpy()))
            print(f"Step {int(step_var.numpy())}: loss={float(loss):.6f}; saved {saved}")

        if batches % 50 == 0:
            print(f"  Batch {batches}; running loss={total_loss / batches:.6f}")

    train_loss = total_loss / max(batches, 1)
    val_loss = evaluate(val_dataset)
    elapsed = time.time() - start
    lr = float(optimizer.learning_rate.numpy())

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["learning_rate"].append(lr)
    history["epoch_seconds"].append(elapsed)

    print(f"Train loss: {train_loss:.6f}")
    print(f"Val loss:   {val_loss:.6f}")
    print(f"LR:         {lr:.8f}")
    print(f"Duration:   {elapsed / 60:.2f} minutes")

    if val_loss < float(best_var.numpy()):
        best_var.assign(val_loss)
        patience_var.assign(0)
        model.save(BEST_MODEL)
        print("New best model saved:", BEST_MODEL)
    else:
        patience_var.assign_add(1)

    epoch_var.assign(epoch_number)
    saved = manager.save(checkpoint_number=int(step_var.numpy()))
    save_history()
    print("Resumable checkpoint:", saved)
    print("History saved:", HISTORY_FILE)

    if int(patience_var.numpy()) >= EARLY_STOPPING_PATIENCE:
        print("Early stopping: validation loss has not improved.")
        break

model.save(FINAL_MODEL)
save_history()

print("\nTRAINING FINISHED")
print("Epochs completed:", int(epoch_var.numpy()))
print("Best validation loss:", float(best_var.numpy()))
print("Best model:", BEST_MODEL.resolve())
print("Final model:", FINAL_MODEL.resolve())
print("Latest checkpoint:", manager.latest_checkpoint)
print("History:", HISTORY_FILE.resolve())
print("To resume, run: python train_v2.py")
