"""
FitnessVision TensorFlow Pose Visualizer
=========================================
Generates visual side-by-side pose estimation plots (Ground Truth vs Prediction)
using TensorFlow / Keras models.

Visual Output Details:
----------------------
- Draws the 16 MPII joint skeleton overlaid on the 256x256 test image.
- Ground truth joints shown in vibrant green; model predictions shown in cyan.
- Computes sample-specific PCK@3 accuracy directly on the figure title.
- Saves high-resolution .png images to training_outputs/gpu_v3_tf/visualizations/.
"""

import sys
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training_gpu.models_tf import build_pose_model
from training_gpu.datasets_tf import load_dataset_records, create_tf_dataset

OUTPUT_DIR = PROJECT_ROOT / "training_outputs" / "gpu_v3_tf"
VIZ_DIR = OUTPUT_DIR / "visualizations"
MODEL_PATH = OUTPUT_DIR / "best_pose_model_tf.keras"
DATASET_DIR = PROJECT_ROOT / "datasets" / "MPII Dataset" / "Full_MPII_Dataset" / "full_16_joint_dataset"

NUM_SAMPLES = 8
HEATMAP_SCALE = 4.0   # 256 / 64 = 4x multiplier to convert heatmap coords to image coords
PCK_THRESHOLD = 3.0   # heatmap pixels

# Standard MPII Skeleton Connections
SKELETON = [
    (0, 1), (1, 2), (2, 6),
    (6, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15)
]


def extract_keypoints_from_heatmaps(heatmaps: np.ndarray) -> list:
    """
    Finds (x, y) coordinates of maximum intensity in each 64x64 heatmap channel.
    Multiplies by HEATMAP_SCALE to project coords onto the original 256x256 image canvas.
    """
    keypoints = []
    for j in range(16):
        hm = heatmaps[:, :, j]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        keypoints.append((float(x * HEATMAP_SCALE), float(y * HEATMAP_SCALE)))
    return keypoints


def draw_pose_skeleton(ax, points: list, mask: np.ndarray, color: str, label: str):
    """
    Draws skeletal limb lines and annotated joint markers on a matplotlib axis.
    """
    for a, b in SKELETON:
        if mask[a] > 0.5 and mask[b] > 0.5:
            ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], color=color, linewidth=2)

    for j, (px, py) in enumerate(points):
        if mask[j] > 0.5:
            ax.scatter(px, py, s=28, color=color, edgecolors="white", linewidths=0.5)
            ax.text(px + 3, py - 3, str(j), fontsize=8, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, lw=0))


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TensorFlow validation dataset...")
    _, val_records = load_dataset_records(DATASET_DIR, val_split=0.2)
    val_ds = create_tf_dataset(val_records, DATASET_DIR, batch_size=8, is_training=False)

    if not MODEL_PATH.exists():
        print(f"Model path {MODEL_PATH} not found. Building untrained model for visual validation...")
        model = build_pose_model()
    else:
        print(f"Loading trained TensorFlow model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    saved_count = 0
    for x_batch, y_batch in val_ds:
        preds = model(x_batch, training=False)

        x_np = x_batch.numpy()
        y_np = y_batch.numpy()
        preds_np = preds.numpy()

        batch_size = x_np.shape[0]

        for i in range(batch_size):
            if saved_count >= NUM_SAMPLES:
                break

            # De-normalize image from [-1, 1] back to [0, 1] for display
            img_disp = (x_np[i] + 1.0) / 2.0
            img_disp = np.clip(img_disp, 0.0, 1.0)

            # Ground truth heatmaps and mask
            gt_heatmaps = y_np[i, :, :, :16]
            mask = y_np[i, 0, 0, 16:]
            pred_heatmaps = preds_np[i]

            gt_pts = extract_keypoints_from_heatmaps(gt_heatmaps)
            pr_pts = extract_keypoints_from_heatmaps(pred_heatmaps)

            # Evaluate sample PCK
            correct = 0
            total = 0
            for j in range(16):
                if mask[j] > 0.5:
                    total += 1
                    dist = np.hypot(gt_pts[j][0] - pr_pts[j][0], gt_pts[j][1] - pr_pts[j][1]) / HEATMAP_SCALE
                    if dist <= PCK_THRESHOLD:
                        correct += 1

            sample_pck = (correct / total * 100.0) if total > 0 else 0.0

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(img_disp)
            draw_pose_skeleton(axes[0], gt_pts, mask, color="lime", label="Ground Truth")
            axes[0].set_title(f"Ground Truth ({total} visible joints)")
            axes[0].axis("off")

            axes[1].imshow(img_disp)
            draw_pose_skeleton(axes[1], pr_pts, mask, color="cyan", label="TensorFlow Prediction")
            axes[1].set_title(f"TensorFlow Prediction (Sample PCK@3: {sample_pck:.1f}%)")
            axes[1].axis("off")

            fig.suptitle(f"FitnessVision TF V3 Model - Validation Sample #{saved_count+1}")
            fig.tight_layout()

            out_file = VIZ_DIR / f"tf_sample_{saved_count+1:02d}.png"
            fig.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

            saved_count += 1
            print(f"Saved: {out_file.name} (Sample PCK@3: {sample_pck:.1f}%)")

        if saved_count >= NUM_SAMPLES:
            break

    print(f"\nCompleted! Generated {saved_count} visualizations in: {VIZ_DIR}")


if __name__ == "__main__":
    main()
