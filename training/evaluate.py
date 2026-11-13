"""
FitnessVision - Pose Estimation Evaluation
Evaluates best_pose_model.keras on the validation dataset using fixed-distance PCK.
Threshold is in 64x64 heatmap pixels. This is not official MPII PCKh.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf

from data_loader import val_dataset, NUM_JOINTS

MODEL_PATH = Path("training_outputs") / "best_pose_model.keras"
PCK_THRESHOLD = 3.0

JOINT_NAMES = [
    "right_ankle", "right_knee", "right_hip", "left_hip",
    "left_knee", "left_ankle", "pelvis", "thorax",
    "upper_neck", "head_top", "right_wrist", "right_elbow",
    "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"
]

# Load the best checkpoint.
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}. "
        "Check that training_outputs/best_pose_model.keras exists."
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

expected_shape = (None, 64, 64, NUM_JOINTS)
if tuple(model.output_shape) != expected_shape:
    raise ValueError(
        f"Unexpected model output shape {model.output_shape}; "
        f"expected {expected_shape}."
    )

print("Loaded model:", MODEL_PATH.resolve())
print("PCK threshold:", PCK_THRESHOLD, "heatmap pixels")

# Accumulate counts and distances independently for each joint.
correct_per_joint = np.zeros(NUM_JOINTS, dtype=np.int64)
total_per_joint = np.zeros(NUM_JOINTS, dtype=np.int64)
distance_sum_per_joint = np.zeros(NUM_JOINTS, dtype=np.float64)

for images, true_heatmaps, joint_masks in val_dataset:
    predictions = model(images, training=False).numpy()
    true_heatmaps = true_heatmaps.numpy()
    joint_masks = joint_masks.numpy()

    for sample_index in range(images.shape[0]):
        for joint_index in range(NUM_JOINTS):
            # Only visibility == 1 joints are included by data_loader.py.
            if joint_masks[sample_index, joint_index] != 1:
                continue

            pred_map = predictions[sample_index, :, :, joint_index]
            true_map = true_heatmaps[sample_index, :, :, joint_index]

            pred_y, pred_x = np.unravel_index(
                np.argmax(pred_map), pred_map.shape
            )
            true_y, true_x = np.unravel_index(
                np.argmax(true_map), true_map.shape
            )

            distance = np.sqrt(
                (pred_x - true_x) ** 2 + (pred_y - true_y) ** 2
            )

            total_per_joint[joint_index] += 1
            distance_sum_per_joint[joint_index] += distance

            if distance <= PCK_THRESHOLD:
                correct_per_joint[joint_index] += 1

# Print overall and per-joint metrics.
total_visible = int(total_per_joint.sum())
total_correct = int(correct_per_joint.sum())

if total_visible == 0:
    raise RuntimeError(
        "No visible validation keypoints were evaluated. "
        "Check the validation data and visibility masks."
    )

overall_pck = 100.0 * total_correct / total_visible

print("\n" + "=" * 58)
print("FITNESSVISION - VALIDATION PCK RESULTS")
print("=" * 58)
print("Visible keypoints evaluated:", total_visible)
print("Correct keypoints:", total_correct)
print("Incorrect keypoints:", total_visible - total_correct)
print(f"Overall PCK: {overall_pck:.2f}%")

print("\nPer-joint results:")
print("-" * 58)

for index, name in enumerate(JOINT_NAMES):
    count = int(total_per_joint[index])
    if count == 0:
        print(f"{name:18s}: No visible samples")
        continue

    pck = 100.0 * correct_per_joint[index] / count
    mean_distance = distance_sum_per_joint[index] / count

    print(
        f"{name:18s}: PCK {pck:6.2f}% | "
        f"{correct_per_joint[index]}/{count} correct | "
        f"mean error {mean_distance:.2f} heatmap pixels"
    )

print("\nNote: This is fixed-threshold PCK on your validation split,")
print("not the official MPII PCKh benchmark.")
print("Evaluation completed.")
