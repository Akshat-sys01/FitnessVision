"""
Training Script for Exercise Pose Estimation CNN
Trains the neural network to output 16 2D joint coordinates from exercise images.
Saves model checkpoints, evaluation metrics, and training loss curves.
"""

import os
import json
import argparse
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import load_annotations, create_tf_dataset
from model import get_model
from loss import masked_huber_loss, masked_mse_loss, PCKMetric


def train_pose_model(
    annotations_json='../filtered_exercise_dataset/exercise_annotations.json',
    images_dir='../filtered_exercise_dataset/images',
    output_dir='checkpoints',
    model_type='custom',
    img_size=(224, 224),
    batch_size=16,
    epochs=10,
    learning_rate=1e-3,
    val_split=0.15
):
    os.makedirs(output_dir, exist_ok=True)
    
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_224')
    print(f"Loading annotations from: {annotations_json}...")
    samples = load_annotations(annotations_json, images_dir, cache_dir=cache_dir)
    print(f"Loaded {len(samples)} valid exercise images with keypoints.")
    
    if len(samples) == 0:
        raise ValueError("No valid samples loaded. Check image and annotation paths.")
        
    # 2. Train / Validation split
    random.seed(42)
    random.shuffle(samples)
    val_count = int(len(samples) * val_split)
    train_samples = samples[val_count:]
    val_samples = samples[:val_count]
    print(f"Dataset split: {len(train_samples)} training samples, {len(val_samples)} validation samples.")
    
    # 3. Create tf.data pipelines
    print("Building data pipelines...")
    train_ds = create_tf_dataset(train_samples, img_size=img_size, batch_size=batch_size, is_training=True)
    val_ds = create_tf_dataset(val_samples, img_size=img_size, batch_size=batch_size, is_training=False)
    
    # 4. Build Model
    print(f"Building {model_type} CNN model...")
    model = get_model(model_name=model_type, input_shape=(img_size[0], img_size[1], 3), num_joints=16)
    model.summary()
    
    # 5. Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=masked_huber_loss,
        metrics=[PCKMetric(threshold=0.08, name="pck_accuracy")]
    )
    
    # 6. Callbacks
    best_model_path = os.path.join(output_dir, "best_pose_cnn.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_pck_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 7. Train
    print(f"\nStarting training for {epochs} epochs (Batch Size: {batch_size})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # 8. Save final model
    final_model_path = os.path.join(output_dir, "final_pose_cnn.keras")
    model.save(final_model_path)
    print(f"\nModels successfully saved to:")
    print(f"  - Best Checkpoint: {best_model_path}")
    print(f"  - Final Model:     {final_model_path}")
    
    # 9. Plot and save training history
    plot_path = os.path.join(output_dir, "training_curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Train Loss', color='#2563EB', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', color='#DC2626', linewidth=2)
    ax1.set_title('Masked Huber Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PCK Accuracy plot
    ax2.plot(history.history['pck_accuracy'], label='Train PCK@0.08', color='#059669', linewidth=2)
    ax2.plot(history.history['val_pck_accuracy'], label='Val PCK@0.08', color='#D97706', linewidth=2)
    ax2.set_title('Joint Localization Accuracy (PCK@0.08)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Training curves saved to: {plot_path}")
    
    # Save training history summary JSON
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'model_type': model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
        }, f, indent=2)
    print(f"Training summary saved to: {summary_path}")
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN for Exercise Joint Prediction.")
    parser.add_argument('--annotations', default='../filtered_exercise_dataset/exercise_annotations.json')
    parser.add_argument('--images', default='../filtered_exercise_dataset/images')
    parser.add_argument('--output', default='checkpoints')
    parser.add_argument('--model-type', choices=['fast', 'custom', 'mobilenet'], default='fast')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    anno_path = os.path.normpath(os.path.join(script_dir, args.annotations))
    img_path = os.path.normpath(os.path.join(script_dir, args.images))
    out_path = os.path.normpath(os.path.join(script_dir, args.output))
    
    train_pose_model(
        annotations_json=anno_path,
        images_dir=img_path,
        output_dir=out_path,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
