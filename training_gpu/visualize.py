"""
Generate Visualizations for Trained FitnessVision GPU V3 Model
Produces side-by-side pose comparisons (Ground Truth vs Prediction)
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training_gpu.models import FitnessVisionPoseNet
from training_gpu.datasets import create_dataloaders

OUTPUT_DIR = PROJECT_ROOT / "training_outputs" / "gpu_v3"
VIZ_DIR = OUTPUT_DIR / "visualizations"
MODEL_PATH = OUTPUT_DIR / "best_pose_model.pth"
DATASET_DIR = PROJECT_ROOT / "datasets" / "MPII Dataset" / "Full_MPII_Dataset" / "full_16_joint_dataset"

NUM_SAMPLES = 8
HEATMAP_SCALE = 4.0
PCK_THRESHOLD = 3.0

JOINT_NAMES = [
    "R ankle", "R knee", "R hip",
    "L hip", "L knee", "L ankle",
    "Pelvis", "Thorax", "Upper neck", "Head top",
    "R wrist", "R elbow", "R shoulder",
    "L shoulder", "L elbow", "L wrist"
]

SKELETON = [
    (0, 1), (1, 2), (2, 6),
    (6, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15)
]


def extract_keypoints(heatmaps):
    points = []
    for j in range(16):
        hm = heatmaps[j]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        points.append((x * HEATMAP_SCALE, y * HEATMAP_SCALE))
    return points


def draw_skeleton(ax, points, masks, color, label):
    for a, b in SKELETON:
        if masks[a] > 0.5 and masks[b] > 0.5:
            ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], color=color, linewidth=2)

    for j, pt in enumerate(points):
        if masks[j] > 0.5:
            ax.scatter(pt[0], pt[1], s=24, color=color, edgecolors="white", linewidths=0.5)
            ax.text(pt[0] + 3, pt[1] - 3, str(j), fontsize=8, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, lw=0))


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = FitnessVisionPoseNet(num_joints=16, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Get Val Loader
    _, val_loader, _, _ = create_dataloaders(DATASET_DIR, batch_size=8, val_split=0.2)

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    saved_count = 0
    with torch.no_grad():
        for images, targets, masks in val_loader:
            images = images.to(device)
            preds = model(images)

            images_np = images.cpu().numpy()
            targets_np = targets.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for i in range(images_np.shape[0]):
                if saved_count >= NUM_SAMPLES:
                    break

                # De-normalize image
                img_display = (images_np[i] * std + mean).transpose(1, 2, 0)
                img_display = np.clip(img_display, 0.0, 1.0)

                gt_pts = extract_keypoints(targets_np[i])
                pr_pts = extract_keypoints(preds_np[i])
                mask = masks_np[i]

                # Sample accuracy
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

                axes[0].imshow(img_display)
                draw_skeleton(axes[0], gt_pts, mask, "lime", "Ground Truth")
                axes[0].set_title(f"Ground Truth ({total} visible joints)")
                axes[0].axis("off")

                axes[1].imshow(img_display)
                draw_skeleton(axes[1], pr_pts, mask, "cyan", "Model Prediction")
                axes[1].set_title(f"Prediction (Sample PCK@3: {sample_pck:.1f}%)")
                axes[1].axis("off")

                fig.suptitle(f"FitnessVision V3 GPU Model - Validation Sample #{saved_count+1}")
                fig.tight_layout()

                save_path = VIZ_DIR / f"v3_sample_{saved_count+1:02d}.png"
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                saved_count += 1
                print(f"Saved visualization: {save_path.name} (Sample PCK: {sample_pck:.1f}%)")

            if saved_count >= NUM_SAMPLES:
                break

    print(f"Finished generating {saved_count} sample visualizations in {VIZ_DIR}")


if __name__ == "__main__":
    main()
