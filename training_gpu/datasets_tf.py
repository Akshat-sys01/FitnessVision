"""
FitnessVision TensorFlow Data Pipeline with Online Augmentation
================================================================
This module streams training and validation batches from disk to GPU
with real-time (online) data augmentation.

How Online Augmentation Works Here:
-----------------------------------
1. Zero Extra Disk Space:
   Images stay untouched on disk. Augmentation happens dynamically in RAM
   only during training, generating fresh variations every epoch.

2. Simple & Explainable Transforms:
   a. Random Horizontal Flip (50% chance):
      - Flips the image horizontally (mirror view).
      - Flips the target heatmaps horizontally.
      - Swaps the left and right joint channels (e.g., Left Wrist <-> Right Wrist).
   b. Random Lighting Variation:
      - Gently adjusts image brightness (+/- 15%) to simulate different workout room lighting.
      - Heatmap coordinates do not change for lighting adjustments.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Generator
import json
import numpy as np
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 256
HEATMAP_SIZE = 64
NUM_JOINTS = 16

# MPII 16 Joint Index Mapping:
# 0: Right Ankle,    1: Right Knee,     2: Right Hip,     3: Left Hip,
# 4: Left Knee,      5: Left Ankle,     6: Pelvis,        7: Thorax,
# 8: Upper Neck,     9: Head Top,      10: Right Wrist,  11: Right Elbow,
# 12: Right Shoulder,13: Left Shoulder, 14: Left Elbow,  15: Left Wrist

# When a human body is mirrored horizontally in a photo, left and right swap:
# - Right Ankle (0) <-> Left Ankle (5)
# - Right Knee (1)  <-> Left Knee (4)
# - Right Hip (2)   <-> Left Hip (3)
# - Central joints (6, 7, 8, 9) stay in the center
# - Right Wrist (10) <-> Left Wrist (15)
# - Right Elbow (11) <-> Left Elbow (14)
# - Right Shoulder (12) <-> Left Shoulder (13)
FLIP_SWAP_MAP = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]

# For the packed tensor (first 16 = heatmaps, next 16 = masks):
PACKED_SWAP_MAP = FLIP_SWAP_MAP + [idx + 16 for idx in FLIP_SWAP_MAP]


def load_dataset_records(
    dataset_dir: Path,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Reads annotations.json and splits the records into training (80%) and validation (20%).
    """
    ann_path = dataset_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("persons", data) if isinstance(data, dict) else data

    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    split_point = int(len(records) * (1.0 - val_split))
    train_records = [records[i] for i in indices[:split_point]]
    val_records = [records[i] for i in indices[split_point:]]

    return train_records, val_records


class MPIITFDataGenerator:
    """
    Streams individual sample files from disk into memory on demand.
    """
    def __init__(self, records: List[Dict[str, Any]], dataset_dir: Path):
        self.records = records
        self.image_dir = dataset_dir / "images"
        self.heatmap_dir = dataset_dir / "heatmaps"

    def __call__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for record in self.records:
            image_name = record["crop_image"]
            stem = Path(image_name).stem

            img_path = self.image_dir / image_name
            npz_path = self.heatmap_dir / f"{stem}.npz"

            # 1. Load image and normalize to [-1, 1]
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                if im.size != (IMAGE_SIZE, IMAGE_SIZE):
                    im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
                image_arr = np.asarray(im, dtype=np.float32)

            image_arr = (image_arr / 127.5) - 1.0

            # 2. Load precomputed heatmaps and visibility mask
            with np.load(npz_path, allow_pickle=False) as npz:
                heatmaps = npz["heatmaps"].astype(np.float32)
                joint_mask = npz["joint_mask"].astype(np.float32)

            # Transpose heatmaps to channels-last: (64, 64, 16)
            hm_channels_last = np.transpose(heatmaps, (1, 2, 0))

            # Broadcast joint mask across spatial area: (64, 64, 16)
            mask_spatial = np.broadcast_to(
                joint_mask.reshape(1, 1, NUM_JOINTS),
                (HEATMAP_SIZE, HEATMAP_SIZE, NUM_JOINTS)
            )

            # Pack heatmaps and masks together into one target: (64, 64, 32)
            packed_target = np.concatenate([hm_channels_last, mask_spatial], axis=-1)

            yield image_arr, packed_target


def augment_sample(image: tf.Tensor, packed_target: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Applies online augmentation during training:
    1. Random Horizontal Flip (50% probability) + Left/Right joint swap
    2. Random Brightness adjustment (+/- 15%)
    """
    # 1. Random Horizontal Flip (50% coin toss)
    if tf.random.uniform([]) > 0.5:
        # Flip image pixels left-to-right
        image = tf.image.flip_left_right(image)

        # Flip target heatmaps and masks left-to-right
        flipped_target = tf.image.flip_left_right(packed_target)

        # Swap left joints with right joints (e.g. Left Wrist <-> Right Wrist)
        packed_target = tf.gather(flipped_target, PACKED_SWAP_MAP, axis=-1)

    # 2. Random Lighting / Brightness Adjustment
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.clip_by_value(image, -1.0, 1.0)

    return image, packed_target


def create_tf_dataset(
    records: List[Dict[str, Any]],
    dataset_dir: Path,
    batch_size: int = 16,
    is_training: bool = True
) -> tf.data.Dataset:
    """
    Builds the tf.data.Dataset pipeline:
    Generator -> (Augment if training) -> Shuffle -> Batch -> Prefetch
    """
    generator = MPIITFDataGenerator(records, dataset_dir)

    output_signature = (
        tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(HEATMAP_SIZE, HEATMAP_SIZE, NUM_JOINTS * 2), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    # Apply online augmentation ONLY on the training split, never validation
    if is_training:
        dataset = dataset.map(augment_sample, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=min(len(records), 512), reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
