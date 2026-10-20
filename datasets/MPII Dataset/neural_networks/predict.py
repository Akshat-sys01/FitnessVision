"""
Inference & Visualization Script for Trained Pose Estimation CNN.
Loads a trained model, predicts the 16 joint coordinates for any input image,
and draws the predicted pose skeleton.
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from loss import masked_huber_loss, masked_mse_loss, PCKMetric

# 16 MPII Joint Names and Skeleton connections
JOINT_NAMES = [
    'r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle',
    'pelvis', 'thorax', 'upper_neck', 'head_top',
    'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist'
]

SKELETON_EDGES = [
    (0, 1),   # r_ankle -> r_knee
    (1, 2),   # r_knee -> r_hip
    (2, 6),   # r_hip -> pelvis
    (3, 6),   # l_hip -> pelvis
    (3, 4),   # l_hip -> l_knee
    (4, 5),   # l_knee -> l_ankle
    (6, 7),   # pelvis -> thorax
    (7, 8),   # thorax -> upper_neck
    (8, 9),   # upper_neck -> head_top
    (7, 12),  # thorax -> r_shoulder
    (12, 11), # r_shoulder -> r_elbow
    (11, 10), # r_elbow -> r_wrist
    (7, 13),  # thorax -> l_shoulder
    (13, 14), # l_shoulder -> l_elbow
    (14, 15)  # l_elbow -> l_wrist
]

RIGHT_JOINTS = {0, 1, 2, 10, 11, 12}
LEFT_JOINTS = {3, 4, 5, 13, 14, 15}


def predict_and_visualize(model_path, image_path, output_path="prediction_output.jpg"):
    """
    Runs model inference on image_path and visualizes predicted joint coordinates.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at: {image_path}")
        
    print(f"Loading trained model from: {model_path}...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'masked_huber_loss': masked_huber_loss,
            'masked_mse_loss': masked_mse_loss,
            'PCKMetric': PCKMetric
        }
    )
    
    # 1. Load original image
    orig_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = orig_img.size
    
    # 2. Preprocess for CNN (224x224, [0, 1])
    resized_img = orig_img.resize((224, 224))
    input_tensor = np.array(resized_img, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0) # Shape: (1, 224, 224, 3)
    
    # 3. Model Inference
    preds = model.predict(input_tensor)[0] # Shape: (32,)
    
    # 4. Denormalize coordinates to original image scale
    pred_points = {}
    print("\nPredicted 2D Joint Coordinates:")
    print("-" * 45)
    for jid, name in enumerate(JOINT_NAMES):
        norm_x = float(preds[jid * 2])
        norm_y = float(preds[jid * 2 + 1])
        px = norm_x * orig_w
        py = norm_y * orig_h
        pred_points[jid] = (px, py)
        print(f"  [{jid:2d}] {name:15s}: ({px:6.1f}, {py:6.1f}) [norm: ({norm_x:.3f}, {norm_y:.3f})]")
    print("-" * 45)
    
    # 5. Draw Skeleton & Joints
    draw = ImageDraw.Draw(orig_img)
    
    # Title Banner
    draw.rectangle([(0, 0), (orig_w, 32)], fill=(0, 0, 0, 200))
    draw.text((12, 8), f"CNN Predicted Pose: {os.path.basename(image_path)}", fill=(0, 255, 255))
    
    # Draw Limbs
    for j1, j2 in SKELETON_EDGES:
        p1 = pred_points[j1]
        p2 = pred_points[j2]
        if j1 in RIGHT_JOINTS and j2 in RIGHT_JOINTS:
            color = (255, 60, 60)    # Red for right side
        elif j1 in LEFT_JOINTS and j2 in LEFT_JOINTS:
            color = (60, 130, 255)   # Blue for left side
        else:
            color = (0, 255, 120)    # Green for spine/center
        draw.line([p1, p2], fill=color, width=4)
        
    # Draw Joint Points
    radius = max(4, int(min(orig_w, orig_h) * 0.008))
    for jid, (px, py) in pred_points.items():
        if jid in RIGHT_JOINTS:
            pt_color = (255, 40, 40)
        elif jid in LEFT_JOINTS:
            pt_color = (40, 100, 255)
        else:
            pt_color = (0, 255, 80)
        draw.ellipse([(px - radius, py - radius), (px + radius, py + radius)], fill=pt_color, outline=(255, 255, 255))
        
    # Save Output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    orig_img.save(output_path)
    print(f"\nPose visualization saved to: {output_path}")
    return pred_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Joint Coordinates with Trained CNN.")
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--model', default='checkpoints/best_pose_cnn.keras', help='Path to trained model .keras')
    parser.add_argument('--output', default='prediction_output.jpg', help='Path to output image')
    args = parser.parse_args()
    
    predict_and_visualize(args.model, args.image, args.output)
