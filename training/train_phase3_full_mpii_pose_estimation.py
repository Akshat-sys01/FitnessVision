"""
FitnessVision V2 training script
Place this file in Full_MPII_Dataset/training/ and run:
    python training/train_fitnessvision.py

Expected data:
    train_validation_dataset/train/images
    train_validation_dataset/train/heatmaps
    train_validation_dataset/train/annotations.json
    train_validation_dataset/validation/images
    train_validation_dataset/validation/heatmaps
    train_validation_dataset/validation/annotations.json

The script uses MobileNetV2 + a lightweight multi-scale decoder.
It writes checkpoints, final model, training history, and a text report.
"""

from pathlib import Path
import json
import os
import time
import traceback

import numpy as np
import tensorflow as tf
from PIL import Image

# This script belongs in Full_MPII_Dataset/training/.
# Dataset splits and training outputs are sibling folders under Full_MPII_Dataset/.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "MPII Dataset"
    / "Full_MPII_Dataset"
    / "train_validation_dataset"
)

OUTPUT_DIR = PROJECT_ROOT / "training_outputs" / "phase3_full_mpii"
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "MPII Dataset"
    / "Full_MPII_Dataset"
    / "train_validation_dataset"
)

OUTPUT_DIR = PROJECT_ROOT / "training_outputs" / "phase3_full_mpii"
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "MPII Dataset"
    / "Full_MPII_Dataset"
    / "train_validation_dataset"
)

OUTPUT_DIR = PROJECT_ROOT / "training_outputs" / "phase3_full_mpii"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_PATH = OUTPUT_DIR / "training_report.txt"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"
MODEL_PATH = OUTPUT_DIR / "fitnessvision_pose_v2.keras"

# Conservative settings for an 8 GB RAM / RTX 2050 laptop.
IMAGE_SIZE = 256
HEATMAP_SIZE = 64
NUM_JOINTS = 16
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42
SIGMA_UNUSED = 2.0

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.keras.utils.set_random_seed(SEED)


def load_records(split):
    annotation_path = DATA_ROOT / split / "annotations.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotations: {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("persons") if isinstance(data, dict) else data
    if not isinstance(records, list) or not records:
        raise ValueError(f"No person records found in {annotation_path}")
    return records


class PoseSequence(tf.keras.utils.Sequence):
    """Loads one batch at a time to avoid holding the dataset in RAM."""

    def __init__(self, records, split, batch_size=BATCH_SIZE, shuffle=False):
        super().__init__()
        self.records = records
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_dir = DATA_ROOT / split / "images"
        self.heatmap_dir = DATA_ROOT / split / "heatmaps"
        self.indices = np.arange(len(records))
        self.rng = np.random.default_rng(SEED)
        self.on_epoch_end()

    def __len__(self):
        return (len(self.records) + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, batch_index):
        batch_ids = self.indices[
            batch_index * self.batch_size:(batch_index + 1) * self.batch_size
        ]
        batch_records = [self.records[i] for i in batch_ids]

        images = np.empty((len(batch_records), IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)
        targets = np.empty(
            (len(batch_records), HEATMAP_SIZE, HEATMAP_SIZE, NUM_JOINTS * 2),
            np.float32
        )

        for i, record in enumerate(batch_records):
            image_name = record["crop_image"]
            image_path = self.image_dir / image_name
            heatmap_path = self.heatmap_dir / (Path(image_name).stem + ".npz")

            with Image.open(image_path) as im:
                im = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                image = np.asarray(im, dtype=np.float32)

            with np.load(heatmap_path, allow_pickle=False) as archive:
                heatmaps = archive["heatmaps"].astype(np.float32)
                mask = archive["joint_mask"].astype(np.float32)

            if heatmaps.shape != (NUM_JOINTS, HEATMAP_SIZE, HEATMAP_SIZE):
                raise ValueError(f"Bad heatmap shape in {heatmap_path}: {heatmaps.shape}")
            if mask.shape != (NUM_JOINTS,):
                raise ValueError(f"Bad joint mask shape in {heatmap_path}: {mask.shape}")

            # TensorFlow MobileNetV2 preprocessing expects pixel values in [-1, 1].
            images[i] = image / 127.5 - 1.0

            # Store heatmaps as channels-last, then append each joint's mask
            # as a constant spatial channel for the custom masked loss.
            hm = np.transpose(heatmaps, (1, 2, 0))
            mask_map = np.broadcast_to(
                mask.reshape(1, 1, NUM_JOINTS),
                (HEATMAP_SIZE, HEATMAP_SIZE, NUM_JOINTS)
            )
            targets[i] = np.concatenate([hm, mask_map], axis=-1)

        return images, targets


def masked_heatmap_mse(y_true, y_pred):
    true_heatmaps = y_true[..., :NUM_JOINTS]
    masks = y_true[..., NUM_JOINTS:]
    squared_error = tf.square(true_heatmaps - y_pred) * masks
    numerator = tf.reduce_sum(squared_error, axis=[1, 2, 3])
    denominator = tf.reduce_sum(masks, axis=[1, 2, 3]) * (
        HEATMAP_SIZE * HEATMAP_SIZE
    )
    per_sample = numerator / tf.maximum(denominator, 1.0)
    return tf.reduce_mean(per_sample)


def build_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")

    # Use ImageNet initialization where available. If the pretrained weights
    # cannot be downloaded, stop with a clear message rather than silently
    # training a different model.
    backbone = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights="imagenet"
    )

    # Initially freeze the backbone for a stable, memory-conscious first run.
    backbone.trainable = False

    skip_64 = backbone.get_layer("block_3_expand_relu").output
    skip_32 = backbone.get_layer("block_6_expand_relu").output
    skip_16 = backbone.get_layer("block_13_expand_relu").output
    x = backbone.get_layer("out_relu").output

    def decoder_block(x, skip, filters, name):
        x = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear", name=f"{name}_upsample"
        )(x)
        x = tf.keras.layers.Concatenate(name=f"{name}_concat")([x, skip])
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"{name}_conv1"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"{name}_conv2"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
        return tf.keras.layers.ReLU(name=f"{name}_relu2")(x)

    x = decoder_block(x, skip_16, 256, "decoder_16")
    x = decoder_block(x, skip_32, 128, "decoder_32")
    x = decoder_block(x, skip_64, 64, "decoder_64")
    outputs = tf.keras.layers.Conv2D(
        NUM_JOINTS, 1, activation="sigmoid", name="joint_heatmaps"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="FitnessVision_MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=masked_heatmap_mse
    )
    return model


def evaluate_pck(model, sequence, threshold=3.0):
    """PCK using argmax locations in 64x64 heatmaps, threshold=3 heatmap pixels."""
    correct = 0
    total = 0

    for batch_index in range(len(sequence)):
        images, packed_targets = sequence[batch_index]
        true_hm = packed_targets[..., :NUM_JOINTS]
        masks = packed_targets[:, 0, 0, NUM_JOINTS:] > 0.5
        predicted_hm = model.predict(images, verbose=0)

        for b in range(images.shape[0]):
            for joint in range(NUM_JOINTS):
                if not masks[b, joint]:
                    continue
                true_map = true_hm[b, :, :, joint]
                pred_map = predicted_hm[b, :, :, joint]
                ty, tx = np.unravel_index(np.argmax(true_map), true_map.shape)
                py, px = np.unravel_index(np.argmax(pred_map), pred_map.shape)
                distance = float(np.hypot(tx - px, ty - py))
                correct += distance <= threshold
                total += 1

    return (correct / total * 100.0) if total else float("nan"), total


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    log = []

    def say(message):
        print(message, flush=True)
        log.append(message)

    try:
        train_records = load_records("train")
        val_records = load_records("validation")
        say(f"Training samples: {len(train_records):,}")
        say(f"Validation samples: {len(val_records):,}")
        say(f"TensorFlow version: {tf.__version__}")
        say(f"Visible GPUs: {tf.config.list_physical_devices('GPU')}")
        say(f"Batch size: {BATCH_SIZE}; epochs: {EPOCHS}")

        train_sequence = PoseSequence(train_records, "train", shuffle=True)
        val_sequence = PoseSequence(val_records, "validation", shuffle=False)

        say("Building MobileNetV2 pose model...")
        model = build_model()
        model.summary()

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(CHECKPOINT_DIR / "best_model.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(str(OUTPUT_DIR / "epoch_log.csv"))
        ]

        say("Starting training. Checkpoints will be saved after each best epoch.")
        history = model.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        model.save(MODEL_PATH)
        say(f"Saved final model: {MODEL_PATH}")

        say("Calculating validation PCK@3 heatmap pixels...")
        pck, visible_joint_count = evaluate_pck(model, val_sequence, threshold=3.0)
        elapsed_minutes = (time.time() - start) / 60.0

        history_dict = {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        }
        HISTORY_PATH.write_text(
            json.dumps(history_dict, indent=2), encoding="utf-8"
        )

        best_val_loss = min(history.history.get("val_loss", [float("nan")]))
        say(f"Best validation loss: {best_val_loss:.6f}")
        say(f"Validation PCK@3: {pck:.2f}% ({visible_joint_count:,} visible joint targets)")
        say(f"Elapsed time: {elapsed_minutes:.1f} minutes")
        say("Training status: COMPLETED")

        report = [
            "FitnessVision V2 Training Report",
            "=" * 40,
            f"Training samples: {len(train_records):,}",
            f"Validation samples: {len(val_records):,}",
            f"Input shape: ({IMAGE_SIZE}, {IMAGE_SIZE}, 3)",
            f"Target heatmap shape: ({HEATMAP_SIZE}, {HEATMAP_SIZE}, {NUM_JOINTS})",
            f"Joint supervision: visibility-masked MSE",
            "Architecture: ImageNet MobileNetV2 encoder + multi-scale decoder",
            "Backbone: frozen for this initial training run",
            f"Batch size: {BATCH_SIZE}",
            f"Maximum epochs: {EPOCHS}",
            f"Learning rate: {LEARNING_RATE}",
            f"Best validation loss: {best_val_loss:.6f}",
            f"Validation PCK@3 heatmap pixels: {pck:.2f}%",
            f"Visible validation joint targets: {visible_joint_count:,}",
            f"Elapsed time: {elapsed_minutes:.1f} minutes",
            f"Final model: {MODEL_PATH}",
            f"Best checkpoint: {CHECKPOINT_DIR / 'best_model.keras'}",
            f"History: {HISTORY_PATH}",
            "",
            "Note: PCK@3 uses a fixed 3-pixel threshold on the 64x64 heatmap grid.",
            "Compare this metric with prior runs only when preprocessing and evaluation",
            "definitions are identical.",
        ]
        REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
        say(f"Report saved: {REPORT_PATH}")

    except Exception as error:
        elapsed_minutes = (time.time() - start) / 60.0
        failure_report = [
            "FitnessVision V2 Training Report",
            "=" * 40,
            "Training status: FAILED",
            f"Error: {error}",
            f"Elapsed time: {elapsed_minutes:.1f} minutes",
            "",
            "Full traceback:",
            traceback.format_exc()
        ]
        REPORT_PATH.write_text("\n".join(failure_report), encoding="utf-8")
        print("\n".join(failure_report), flush=True)
        raise


if __name__ == "__main__":
    main()
