"""
FitnessVision TensorFlow Data Pipeline with Online Augmentation
================================================================
This module streams training, validation, and test batches from disk to GPU
with real-time (online) data augmentation.

Supports both:
1. Unified Pose Dataset (MPII + COCO 80k standardized format with online heatmap generation)
2. Legacy precomputed .npz heatmaps (for backward compatibility)

How Online Augmentation Works:
------------------------------
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
SIGMA = 2.0  # Gaussian heatmap standard deviation in 64x64 resolution

# MPII / Canonical 16 Joint Index Mapping:
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


def load_unified_dataset_records(dataset_dir: Path, split: str = "train") -> List[Dict[str, Any]]:
    """
    Loads unified dataset records directly from the split JSON file:
    splits/train.json, splits/validation.json, or splits/test.json.
    """
    split_file = dataset_dir / "splits" / f"{split}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with split_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("items", data)


def load_dataset_records(
    dataset_dir: Path,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads dataset records. Automatically detects whether the directory is:
    - Unified_Pose_Dataset (has splits/ directory)
    - Legacy single-annotation directory (annotations.json)
    """
    splits_dir = dataset_dir / "splits"
    if splits_dir.exists() and (splits_dir / "train.json").exists():
        train_records = load_unified_dataset_records(dataset_dir, split="train")
        val_records = load_unified_dataset_records(dataset_dir, split="validation")
        return train_records, val_records

    # Legacy fallback
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


def generate_gaussian_heatmap(
    keypoints: List[Any],
    training_mask: List[int],
    orig_w: int,
    orig_h: int,
    heatmap_size: int = HEATMAP_SIZE,
    sigma: float = SIGMA
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates 2D Gaussian heatmaps (64x64x16) and spatial masks (64x64x16)
    from canonical keypoints list [ [x, y] or None, ... ].
    """
    heatmaps = np.zeros((heatmap_size, heatmap_size, NUM_JOINTS), dtype=np.float32)
    mask_spatial = np.zeros((heatmap_size, heatmap_size, NUM_JOINTS), dtype=np.float32)

    scale_x = heatmap_size / max(float(orig_w), 1.0)
    scale_y = heatmap_size / max(float(orig_h), 1.0)

    # 2D coordinate grid
    grid_y, grid_x = np.ogrid[0:heatmap_size, 0:heatmap_size]
    two_sigma_sq = 2.0 * sigma * sigma

    for j in range(NUM_JOINTS):
        if training_mask[j] == 1 and keypoints[j] is not None:
            kx, ky = keypoints[j]
            hm_x = kx * scale_x
            hm_y = ky * scale_y

            # Check within heatmap bounds
            if 0 <= hm_x < heatmap_size and 0 <= hm_y < heatmap_size:
                dist_sq = (grid_x - hm_x) ** 2 + (grid_y - hm_y) ** 2
                heatmaps[:, :, j] = np.exp(-dist_sq / two_sigma_sq)
                mask_spatial[:, :, j] = 1.0

    return heatmaps, mask_spatial


class UnifiedTFDataGenerator:
    """
    Streams samples from Unified_Pose_Dataset (or legacy datasets):
    - Loads image and resizes to 256x256
    - Generates 64x64 Gaussian heatmaps online (or loads .npz if available)
    - Packs heatmaps (16 channels) and spatial masks (16 channels) into (64, 64, 32)
    """
    def __init__(self, records: List[Dict[str, Any]], dataset_dir: Path):
        self.records = records
        self.image_dir = dataset_dir / "images"
        self.heatmap_dir = dataset_dir / "heatmaps"
        self.has_precomputed_heatmaps = self.heatmap_dir.exists()

    def __call__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        for record in self.records:
            # Check filename attribute
            image_name = record.get("file_name") or record.get("crop_image")
            stem = Path(image_name).stem
            img_path = self.image_dir / image_name

            # 1. Load image and normalize to [-1, 1]
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    orig_w, orig_h = im.size
                    if (orig_w, orig_h) != (IMAGE_SIZE, IMAGE_SIZE):
                        im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
                    image_arr = np.asarray(im, dtype=np.float32)
            except Exception:
                continue

            image_arr = (image_arr / 127.5) - 1.0

            # 2. Obtain heatmaps and masks
            npz_path = self.heatmap_dir / f"{stem}.npz" if self.has_precomputed_heatmaps else None
            if npz_path and npz_path.exists():
                with np.load(npz_path, allow_pickle=False) as npz:
                    heatmaps = npz["heatmaps"].astype(np.float32)
                    joint_mask = npz["joint_mask"].astype(np.float32)

                hm_channels_last = np.transpose(heatmaps, (1, 2, 0))
                mask_spatial = np.broadcast_to(
                    joint_mask.reshape(1, 1, NUM_JOINTS),
                    (HEATMAP_SIZE, HEATMAP_SIZE, NUM_JOINTS)
                )
            else:
                # Online Gaussian heatmap generation
                kps = record.get("keypoints", [])
                tmask = record.get("training_mask", [0] * NUM_JOINTS)
                w = record.get("width", orig_w)
                h = record.get("height", orig_h)
                hm_channels_last, mask_spatial = generate_gaussian_heatmap(kps, tmask, w, h)

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
        image = tf.image.flip_left_right(image)
        flipped_target = tf.image.flip_left_right(packed_target)
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
    generator = UnifiedTFDataGenerator(records, dataset_dir)

    output_signature = (
        tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(HEATMAP_SIZE, HEATMAP_SIZE, NUM_JOINTS * 2), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    # Apply online augmentation ONLY on the training split, never validation/test
    if is_training:
        dataset = dataset.map(augment_sample, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=min(len(records), 512), reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
