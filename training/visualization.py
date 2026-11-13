
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import val_dataset, NUM_JOINTS


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path("training_outputs/best_pose_model.keras")
OUTPUT_DIR = Path("training_outputs/visualizations")

NUM_SAMPLES = 4
HEATMAP_SCALE = 4  # 64x64 heatmap -> 256x256 image


JOINT_NAMES = [
    "Right ankle", "Right knee", "Right hip",
    "Left hip", "Left knee", "Left ankle",
    "Pelvis", "Thorax", "Upper neck", "Head top",
    "Right wrist", "Right elbow", "Right shoulder",
    "Left shoulder", "Left elbow", "Left wrist"
]

# MPII skeleton connections
SKELETON = [
    (0, 1), (1, 2), (2, 6),
    (6, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15)
]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Loaded model:", MODEL_PATH.resolve())


# --------------------------------------------------
# KEYPOINT EXTRACTION
# --------------------------------------------------

def extract_keypoints(heatmaps):
    """
    Extract (x, y) coordinates from 16 heatmaps.
    Returns coordinates in 64x64 heatmap space.
    """

    keypoints = []

    for joint_index in range(NUM_JOINTS):
        heatmap = heatmaps[:, :, joint_index]

        y, x = np.unravel_index(
            np.argmax(heatmap),
            heatmap.shape
        )

        keypoints.append((x, y))

    return np.array(keypoints)


# --------------------------------------------------
# DRAW SKELETON
# --------------------------------------------------

def draw_pose(ax, image, keypoints, joint_mask, title):
    ax.imshow(image)

    # Draw connections first
    for joint_a, joint_b in SKELETON:

        if joint_mask[joint_a] != 1:
            continue

        if joint_mask[joint_b] != 1:
            continue

        x1, y1 = keypoints[joint_a] * HEATMAP_SCALE
        x2, y2 = keypoints[joint_b] * HEATMAP_SCALE

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=2
        )

    # Draw keypoints
    for index, (x, y) in enumerate(keypoints):

        if joint_mask[index] != 1:
            continue

        x *= HEATMAP_SCALE
        y *= HEATMAP_SCALE

        ax.scatter(x, y, s=25)
        ax.text(
            x + 2,
            y - 2,
            str(index),
            fontsize=7
        )

    ax.set_title(title)
    ax.axis("off")


# --------------------------------------------------
# VISUALIZE VALIDATION SAMPLES
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sample_count = 0

for images, true_heatmaps, joint_masks in val_dataset:

    predictions = model(
        images,
        training=False
    ).numpy()

    images = images.numpy()
    true_heatmaps = true_heatmaps.numpy()
    joint_masks = joint_masks.numpy()

    for i in range(images.shape[0]):

        # Skip samples without visible joints
        if np.sum(joint_masks[i]) == 0:
            continue

        true_keypoints = extract_keypoints(
            true_heatmaps[i]
        )

        predicted_keypoints = extract_keypoints(
            predictions[i]
        )

        fig, axes = plt.subplots(
            1, 2,
            figsize=(12, 6)
        )

        draw_pose(
            axes[0],
            images[i],
            true_keypoints,
            joint_masks[i],
            "GROUND TRUTH"
        )

        draw_pose(
            axes[1],
            images[i],
            predicted_keypoints,
            joint_masks[i],
            "MODEL PREDICTION"
        )

        fig.suptitle(
            f"Validation Sample {sample_count + 1}",
            fontsize=14
        )

        plt.tight_layout()

        save_path = (
            OUTPUT_DIR /
            f"sample_{sample_count + 1}.png"
        )

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        print("Saved:", save_path)

        plt.show()
        plt.close(fig)

        sample_count += 1

        if sample_count >= NUM_SAMPLES:
            break

    if sample_count >= NUM_SAMPLES:
        break


print("\nVisualization completed.")
print("Total samples visualized:", sample_count)
print("Output directory:", OUTPUT_DIR.resolve())
