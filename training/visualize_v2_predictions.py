"""
Visualize FitnessVision V2 predictions against ground truth.

Run from C:\\FitnessVision:
    python visualize_v2_predictions.py

Uses the same validation dataset and heatmap-argmax decoding as evaluate.py.
Saves side-by-side overlays in training_outputs_v2/visualizations/.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import val_dataset, NUM_JOINTS


MODEL_PATH = Path("training_outputs_v2/best_pose_model_v2.keras")
OUTPUT_DIR = Path("training_outputs_v2/visualizations")

NUM_SAMPLES = 8
HEATMAP_SCALE = 4

JOINT_NAMES = [
    "R ankle", "R knee", "R hip",
    "L hip", "L knee", "L ankle",
    "Pelvis", "Thorax", "Upper neck", "Head top",
    "R wrist", "R elbow", "R shoulder",
    "L shoulder", "L elbow", "L wrist"
]

# MPII skeleton connections using the joint order above.
SKELETON = [
    (0, 1), (1, 2), (2, 6),
    (6, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15)
]


def unpack_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise ValueError(
            "Expected val_dataset batches as "
            "(images, heatmaps, joint_masks)."
        )

    images, heatmaps, masks = batch
    images = tf.cast(images, tf.float32)
    heatmaps = tf.cast(heatmaps, tf.float32)
    masks = tf.cast(masks, tf.float32)

    # Convert [B, 16, 64, 64] to [B, 64, 64, 16], if needed.
    if heatmaps.shape.rank == 4 and heatmaps.shape[1] == NUM_JOINTS:
        heatmaps = tf.transpose(heatmaps, [0, 2, 3, 1])

    if masks.shape.rank == 3 and masks.shape[-1] == 1:
        masks = tf.squeeze(masks, axis=-1)

    return images.numpy(), heatmaps.numpy(), masks.numpy()


def extract_keypoints(heatmaps):
    """Return argmax (x,y) for each heatmap in 64x64 coordinates."""
    points = []

    for joint_index in range(NUM_JOINTS):
        heatmap = heatmaps[:, :, joint_index]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        points.append((x, y))

    return np.asarray(points, dtype=np.float32)


def draw_pose(ax, image, points, visible_mask, title):
    ax.imshow(np.clip(image, 0.0, 1.0))

    # Draw only connections whose two joints have ground-truth visibility.
    for a, b in SKELETON:
        if visible_mask[a] != 1 or visible_mask[b] != 1:
            continue

        xa, ya = points[a] * HEATMAP_SCALE
        xb, yb = points[b] * HEATMAP_SCALE
        ax.plot([xa, xb], [ya, yb], linewidth=2)

    for joint_id, (x, y) in enumerate(points):
        if visible_mask[joint_id] != 1:
            continue

        x *= HEATMAP_SCALE
        y *= HEATMAP_SCALE
        ax.scatter(x, y, s=24)
        ax.text(x + 2, y - 2, str(joint_id), fontsize=7)

    ax.set_title(title)
    ax.axis("off")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find V2 model: {MODEL_PATH.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading:", MODEL_PATH.resolve())
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model output:", model.output_shape)

    sample_number = 0

    for batch in val_dataset:
        images, true_heatmaps, masks = unpack_batch(batch)
        predictions = model(images, training=False).numpy()

        for i in range(len(images)):
            visible = masks[i] > 0.5

            if not np.any(visible):
                continue

            gt_points = extract_keypoints(true_heatmaps[i])
            pred_points = extract_keypoints(predictions[i])

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            draw_pose(
                axes[0], images[i], gt_points, visible,
                "GROUND TRUTH"
            )
            draw_pose(
                axes[1], images[i], pred_points, visible,
                "V2 PREDICTION"
            )

            fig.suptitle(
                f"FitnessVision V2 - Validation sample {sample_number + 1}"
            )
            fig.tight_layout()

            output_path = OUTPUT_DIR / f"v2_sample_{sample_number + 1:02d}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print("Saved:", output_path)

            plt.show()
            plt.close(fig)

            sample_number += 1
            if sample_number >= NUM_SAMPLES:
                print(f"\nCreated {sample_number} visualizations.")
                print("Output folder:", OUTPUT_DIR.resolve())
                return

    print(f"\nDataset ended. Created {sample_number} visualizations.")
    print("Output folder:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
