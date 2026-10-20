"""
CNN Architectures for 2D Human Pose Joint Coordinate Regression.
Provides:
1. FastPoseCNN: A lightweight, mobile/CPU-optimized depthwise separable CNN (default).
2. CustomPoseCNN: A multi-stage deep convolutional neural network for GPU training.
3. MobileNetPoseCNN: Transfer learning model based on MobileNetV2.
All architectures output 32 values representing the (x, y) coordinates for 16 body joints.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_fast_pose_cnn(input_shape=(224, 224, 3), num_joints=16):
    inputs = layers.Input(shape=input_shape, name="image_input")

    # --- Encoder (same as before) ---
    x = layers.Conv2D(16, 3, strides=2, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.SeparableConv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.SeparableConv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.SeparableConv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    # --- Keep spatial feature map (14×14×128), don't GAP yet ---
    # Instead: use a lightweight heatmap-style head

    # Project to num_joints channels → each channel = one joint's spatial response
    heatmaps = layers.Conv2D(num_joints, 1, padding="same", name="heatmap_conv")(x)  # (B, 14, 14, 16)

    # Soft-argmax: weighted average of (x, y) grid positions per joint
    outputs = soft_argmax_2d(heatmaps, num_joints)  # (B, 32)

    return models.Model(inputs=inputs, outputs=outputs, name="FastPoseCNN_v2")

def soft_argmax_2d(heatmaps, num_joints):
    """
    Differentiable spatial argmax. Converts heatmap (B,H,W,J) → coordinates (B, J*2).
    Each joint gets its own spatial probability map → naturally outputs different locations.
    """
    shape = tf.shape(heatmaps)
    H, W = shape[1], shape[2]

    # Softmax over spatial dims
    flat = tf.reshape(heatmaps, [-1, H * W, num_joints])
    weights = tf.nn.softmax(flat, axis=1)  # (B, H*W, J)

    # Build normalized coordinate grids
    xs = tf.cast(tf.range(W), tf.float32) / tf.cast(W - 1, tf.float32)
    ys = tf.cast(tf.range(H), tf.float32) / tf.cast(H - 1, tf.float32)
    grid_x, grid_y = tf.meshgrid(xs, ys)
    grid_x = tf.reshape(grid_x, [-1])  # (H*W,)
    grid_y = tf.reshape(grid_y, [-1])

    # Weighted sum → expected coordinates per joint
    pred_x = tf.reduce_sum(weights * grid_x[tf.newaxis, :, tf.newaxis], axis=1)  # (B, J)
    pred_y = tf.reduce_sum(weights * grid_y[tf.newaxis, :, tf.newaxis], axis=1)

    # Interleave x,y → (B, J*2) matching your existing loss format
    pred = tf.stack([pred_x, pred_y], axis=-1)  # (B, J, 2)
    return tf.reshape(pred, [-1, num_joints * 2])
hbb
def build_custom_pose_cnn(input_shape=(224, 224, 3), num_joints=16):
    """
    Builds a deep custom CNN with standard convolutions.
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_joints * 2, activation="sigmoid", name="joint_coordinates")(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="CustomPoseCNN")


def build_mobilenet_pose_cnn(input_shape=(224, 224, 3), num_joints=16):
    """
    Builds a pose regression network using MobileNetV2 backbone.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = True
    
    inputs = layers.Input(shape=input_shape, name="image_input")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="drop1")(x)
    outputs = layers.Dense(num_joints * 2, activation="sigmoid", name="joint_coordinates")(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="MobileNetPoseCNN")


def get_model(model_name="fast", input_shape=(224, 224, 3), num_joints=16):
    name = model_name.lower()
    if name == "fast":
        return build_fast_pose_cnn(input_shape, num_joints)
    elif name == "mobilenet":
        return build_mobilenet_pose_cnn(input_shape, num_joints)
    else:
        return build_custom_pose_cnn(input_shape, num_joints)
