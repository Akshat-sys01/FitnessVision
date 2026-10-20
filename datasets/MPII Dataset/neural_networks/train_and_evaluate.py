"""
Extended Training + Comprehensive Evaluation Script
Trains the FastPoseCNN for more epochs (resuming from saved weights),
then evaluates on the validation set and visualizes predictions on sample images.
"""

import os
import sys
import json
import random
import numpy as np
from PIL import Image, ImageDraw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Add neural_networks dir to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from dataset import load_annotations, create_tf_dataset
from model import get_model
from loss import masked_huber_loss, PCKMetric

# ===================== CONFIG =====================
ANNOTATIONS = os.path.join(script_dir, '..', 'filtered_exercise_dataset', 'exercise_annotations.json')
IMAGES_DIR = os.path.join(script_dir, '..', 'filtered_exercise_dataset', 'images')
CACHE_DIR = os.path.join(script_dir, 'cache_224')
CHECKPOINT_DIR = os.path.join(script_dir, 'checkpoints')
RESULTS_DIR = os.path.join(script_dir, 'results')

EPOCHS = 25
BATCH_SIZE = 32
LR = 5e-4
VAL_SPLIT = 0.15
IMG_SIZE = (224, 224)

JOINT_NAMES = [
    'r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle',
    'pelvis', 'thorax', 'upper_neck', 'head_top',
    'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist'
]

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 6), (3, 6), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15)
]

RIGHT_JOINTS = {0, 1, 2, 10, 11, 12}
LEFT_JOINTS = {3, 4, 5, 13, 14, 15}


def draw_skeleton(img, pred_pts, gt_pts=None, title=""):
    """Draws predicted (and optional ground truth) skeleton on image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Title
    draw.rectangle([(0, 0), (w, 28)], fill=(0, 0, 0))
    draw.text((8, 6), title, fill=(0, 255, 255))

    def draw_pose(pts, color_r, color_l, color_c, radius, width):
        for j1, j2 in SKELETON_EDGES:
            if j1 in pts and j2 in pts:
                if j1 in RIGHT_JOINTS and j2 in RIGHT_JOINTS:
                    c = color_r
                elif j1 in LEFT_JOINTS and j2 in LEFT_JOINTS:
                    c = color_l
                else:
                    c = color_c
                draw.line([pts[j1], pts[j2]], fill=c, width=width)
        for jid, (px, py) in pts.items():
            if jid in RIGHT_JOINTS:
                c = color_r
            elif jid in LEFT_JOINTS:
                c = color_l
            else:
                c = color_c
            draw.ellipse([(px-radius, py-radius), (px+radius, py+radius)], fill=c, outline=(255,255,255))

    # Draw ground truth first (if provided) in semi-transparent green
    if gt_pts:
        draw_pose(gt_pts, (100, 200, 100), (100, 200, 100), (100, 200, 100), 4, 2)

    # Draw predictions in bright colors
    draw_pose(pred_pts, (255, 60, 60), (60, 130, 255), (0, 255, 120), 6, 4)

    return img


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ===================== LOAD DATA =====================
    print("=" * 60)
    print("PHASE 1: LOADING DATA")
    print("=" * 60)
    samples = load_annotations(ANNOTATIONS, IMAGES_DIR, cache_dir=CACHE_DIR)
    print(f"Total valid samples: {len(samples)}")

    random.seed(42)
    random.shuffle(samples)
    val_count = int(len(samples) * VAL_SPLIT)
    train_samples = samples[val_count:]
    val_samples = samples[:val_count]
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_ds = create_tf_dataset(train_samples, img_size=IMG_SIZE, batch_size=BATCH_SIZE, is_training=True)
    val_ds = create_tf_dataset(val_samples, img_size=IMG_SIZE, batch_size=BATCH_SIZE, is_training=False)

    # ===================== BUILD / LOAD MODEL =====================
    print("\n" + "=" * 60)
    print("PHASE 2: BUILDING MODEL")
    print("=" * 60)

    best_path = os.path.join(CHECKPOINT_DIR, 'best_pose_cnn.keras')
    if os.path.exists(best_path):
        print(f"Loading existing model from: {best_path}")
        model = tf.keras.models.load_model(best_path, custom_objects={
            'masked_huber_loss': masked_huber_loss,
            'PCKMetric': PCKMetric
        })
    else:
        print("Building new FastPoseCNN model...")
        model = get_model('fast', input_shape=(224, 224, 3), num_joints=16)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=masked_huber_loss,
        metrics=[PCKMetric(threshold=0.08, name="pck_008"), PCKMetric(threshold=0.15, name="pck_015")]
    )
    model.summary()
    print(f"Total parameters: {model.count_params():,}")

    # ===================== TRAIN =====================
    print("\n" + "=" * 60)
    print(f"PHASE 3: TRAINING FOR {EPOCHS} EPOCHS")
    print("=" * 60)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_path, monitor="val_pck_008", mode="max",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        )
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, 'final_pose_cnn.keras')
    model.save(final_path)

    # ===================== EVALUATE =====================
    print("\n" + "=" * 60)
    print("PHASE 4: EVALUATION ON VALIDATION SET")
    print("=" * 60)

    results = model.evaluate(val_ds, verbose=1)
    metric_names = model.metrics_names
    print("\nValidation Results:")
    for name, val in zip(metric_names, results):
        print(f"  {name}: {val:.4f}")

    # ===================== PER-JOINT ACCURACY =====================
    print("\n" + "=" * 60)
    print("PHASE 5: PER-JOINT PCK ACCURACY")
    print("=" * 60)

    per_joint_correct = np.zeros(16)
    per_joint_total = np.zeros(16)

    for batch_x, batch_y in val_ds:
        preds = model.predict(batch_x, verbose=0)
        kps_true = batch_y[:, :32].numpy()
        mask = batch_y[:, 32:].numpy()

        true_xy = kps_true.reshape(-1, 16, 2)
        pred_xy = preds.reshape(-1, 16, 2)
        mask_xy = mask.reshape(-1, 16, 2)[:, :, 0]

        dists = np.linalg.norm(true_xy - pred_xy, axis=-1)

        for j in range(16):
            valid = mask_xy[:, j] > 0
            per_joint_total[j] += np.sum(valid)
            per_joint_correct[j] += np.sum((dists[:, j] <= 0.08) & valid)

    print(f"\n{'Joint':<15} | {'PCK@0.08':>10} | {'Correct':>8} / {'Total':>8}")
    print("-" * 55)
    for j in range(16):
        pck = per_joint_correct[j] / max(per_joint_total[j], 1) * 100
        print(f"  {JOINT_NAMES[j]:<13} | {pck:9.1f}% | {int(per_joint_correct[j]):>8} / {int(per_joint_total[j]):>8}")
    overall_pck = np.sum(per_joint_correct) / max(np.sum(per_joint_total), 1) * 100
    print("-" * 55)
    print(f"  {'OVERALL':<13} | {overall_pck:9.1f}% | {int(np.sum(per_joint_correct)):>8} / {int(np.sum(per_joint_total)):>8}")

    # ===================== VISUALIZE SAMPLES =====================
    print("\n" + "=" * 60)
    print("PHASE 6: SAMPLE PREDICTIONS & VISUALIZATIONS")
    print("=" * 60)

    target_activities = [
        'calisthenics', 'resistance training', 'weight lifting, power lifting',
        'yoga, Power', 'circuit training', 'home exercise, general',
        'rope skipping, general', 'aerobic, step'
    ]

    picked = {}
    for s in val_samples:
        act = s['activity']
        if act in target_activities and act not in picked:
            picked[act] = s
        if len(picked) >= len(target_activities):
            break

    # Fill remaining with any val samples
    if len(picked) < 8:
        for s in val_samples:
            if s['activity'] not in picked:
                picked[s['activity']] = s
            if len(picked) >= 8:
                break

    for act, sample in picked.items():
        orig_path = os.path.join(IMAGES_DIR, sample['image_name'])
        if not os.path.exists(orig_path):
            continue

        orig_img = Image.open(orig_path).convert('RGB')
        ow, oh = orig_img.size

        resized = orig_img.resize((224, 224))
        inp = np.array(resized, dtype=np.float32) / 255.0
        inp = np.expand_dims(inp, axis=0)

        pred = model.predict(inp, verbose=0)[0]

        pred_pts = {}
        gt_pts = {}
        kps_arr = sample['keypoints']
        mask_arr = sample['mask']

        for j in range(16):
            px = float(pred[j*2]) * ow
            py = float(pred[j*2+1]) * oh
            pred_pts[j] = (px, py)

            if mask_arr[j*2] > 0:
                gx = float(kps_arr[j*2]) * ow
                gy = float(kps_arr[j*2+1]) * oh
                gt_pts[j] = (gx, gy)

        vis_img = draw_skeleton(orig_img.copy(), pred_pts, gt_pts,
                                title=f"Pred(bright) vs GT(green) | {act}")

        safe_name = act.replace(' ', '_').replace(',', '').replace('/', '_')[:40]
        out_path = os.path.join(RESULTS_DIR, f"pred_{safe_name}.jpg")
        vis_img.save(out_path, quality=90)
        print(f"  Saved: {out_path}")

    # ===================== SAVE TRAINING CURVES =====================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history.history['loss'], label='Train Loss', color='#2563EB', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='#DC2626', linewidth=2)
    axes[0].set_title('Masked Huber Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['pck_008'], label='Train PCK@0.08', color='#059669', linewidth=2)
    axes[1].plot(history.history['val_pck_008'], label='Val PCK@0.08', color='#D97706', linewidth=2)
    axes[1].set_title('PCK@0.08 (Strict)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history.history['pck_015'], label='Train PCK@0.15', color='#7C3AED', linewidth=2)
    axes[2].plot(history.history['val_pck_015'], label='Val PCK@0.15', color='#DB2777', linewidth=2)
    axes[2].set_title('PCK@0.15 (Relaxed)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, 'training_curves.png')
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"\nTraining curves saved to: {curves_path}")

    # ===================== SAVE FULL REPORT =====================
    report = {
        'model_type': 'FastPoseCNN',
        'total_params': model.count_params(),
        'epochs_trained': len(history.history['loss']),
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_pck008': float(history.history['pck_008'][-1]),
        'final_val_pck008': float(history.history['val_pck_008'][-1]),
        'final_train_pck015': float(history.history['pck_015'][-1]),
        'final_val_pck015': float(history.history['val_pck_015'][-1]),
        'per_joint_pck008': {
            JOINT_NAMES[j]: round(per_joint_correct[j] / max(per_joint_total[j], 1) * 100, 2)
            for j in range(16)
        },
        'overall_pck008': round(overall_pck, 2),
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }
    report_path = os.path.join(RESULTS_DIR, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Full evaluation report saved to: {report_path}")
    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
