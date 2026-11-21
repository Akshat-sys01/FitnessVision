"""
FitnessVision ResNet-50 Pose Estimation Architecture
====================================================
This module replaces MobileNetV2 with an ImageNet-pretrained ResNet-50 backbone
combined with a multi-scale Feature Pyramid Network (FPN) decoder.

Why ResNet-50 is Superior:
--------------------------
1. High-Capacity Residual Blocks:
   - ResNet-50 uses 25.6M parameters (vs. 3.5M in MobileNetV2), providing deep residual
     features that capture subtle anatomical details (wrists, elbows, ankles).
2. Multi-Scale Skip Fusion:
   - conv2_block3_out -> 64x64 resolution (retains fine boundary details)
   - conv3_block4_out -> 32x32 resolution (captures limb orientations)
   - conv4_block6_out -> 16x16 resolution (captures body posture context)
   - conv5_block3_out -> 8x8 resolution (deep semantic body representation)
3. Joint-Weighted Masked Loss:
   - Wrists and ankles are prioritized with higher loss weights (1.5x and 1.2x)
     to directly fix the historical extremity accuracy bottleneck.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Joint importance weights (MPII 16 joints):
# Wrists (10, 15) get 1.5x weight; Ankles (0, 5) get 1.2x; Torso joints get 0.8x
# This forces the optimizer to pay extra attention to fast-moving extremities.
JOINT_WEIGHTS = [
    1.2,  # 0: Right Ankle
    1.0,  # 1: Right Knee
    0.8,  # 2: Right Hip
    0.8,  # 3: Left Hip
    1.0,  # 4: Left Knee
    1.2,  # 5: Left Ankle
    0.8,  # 6: Pelvis
    0.8,  # 7: Thorax
    0.9,  # 8: Upper Neck
    0.9,  # 9: Head Top
    1.5,  # 10: Right Wrist (High priority!)
    1.1,  # 11: Right Elbow
    0.8,  # 12: Right Shoulder
    0.8,  # 13: Left Shoulder
    1.1,  # 14: Left Elbow
    1.5,  # 15: Left Wrist (High priority!)
]


def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """
    Standard building block: 3x3 Conv -> Batch Normalization -> ReLU.
    use_bias=False because Batch Normalization already includes bias shifts.
    """
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return x


def build_pose_model(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_joints: int = 16
) -> keras.Model:
    """
    Constructs the ResNet-50 Pose Estimation Network.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # 1. Pretrained ResNet-50 Feature Extractor
    base_resnet = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )
    base_resnet.trainable = True  # Fine-tune the backbone end-to-end

    # 2. Extract Multi-Scale Feature Maps
    skip_64 = base_resnet.get_layer("conv2_block3_out").output  # (64x64, 256 ch)
    skip_32 = base_resnet.get_layer("conv3_block4_out").output  # (32x32, 512 ch)
    skip_16 = base_resnet.get_layer("conv4_block6_out").output  # (16x16, 1024 ch)
    deep_8  = base_resnet.get_layer("conv5_block3_out").output  # (8x8, 2048 ch)

    # 3. Progressive Upsampling Decoder
    # 8x8 -> 16x16
    x = conv_block(deep_8, 256, "dec_bottleneck")
    x = layers.UpSampling2D((2, 2), interpolation="bilinear", name="up_16")(x)
    skip_16_proj = layers.Conv2D(256, 1, padding="same", name="proj_16")(skip_16)
    x = layers.Concatenate(axis=-1)([x, skip_16_proj])
    x = conv_block(x, 256, "dec_16_refine")

    # 16x16 -> 32x32
    x = layers.UpSampling2D((2, 2), interpolation="bilinear", name="up_32")(x)
    skip_32_proj = layers.Conv2D(128, 1, padding="same", name="proj_32")(skip_32)
    x = layers.Concatenate(axis=-1)([x, skip_32_proj])
    x = conv_block(x, 128, "dec_32_refine")

    # 32x32 -> 64x64
    x = layers.UpSampling2D((2, 2), interpolation="bilinear", name="up_64")(x)
    skip_64_proj = layers.Conv2D(96, 1, padding="same", name="proj_64")(skip_64)
    x = layers.Concatenate(axis=-1)([x, skip_64_proj])
    x = conv_block(x, 96, "dec_64_refine")

    # 4. Final Heatmap Prediction Head (16 channels at 64x64)
    # Linear activation preserves Gaussian peaks without gradient saturation.
    outputs = layers.Conv2D(
        num_joints,
        kernel_size=1,
        padding="same",
        activation=None,
        name="joint_heatmaps"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="FitnessVision_ResNet50_Pose")
    return model


class MaskedHeatmapMSELoss(keras.losses.Loss):
    """
    Weighted Masked MSE Loss:
    - Ignores invisible / missing joints (mask = 0).
    - Multiplies errors by JOINT_WEIGHTS so wrists and ankles receive higher gradient focus.
    """
    def __init__(self, name: str = "weighted_masked_mse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weights_tensor = tf.constant(JOINT_WEIGHTS, dtype=tf.float32)

    def call(self, y_true_packed: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Separate heatmaps (first 16) and visibility masks (last 16)
        y_true = y_true_packed[..., :16]
        masks = y_true_packed[..., 16:]

        # Element-wise squared error
        sq_err = tf.square(y_true - y_pred) * masks

        # Spatial MSE per joint: (batch, 16)
        mse_per_joint = tf.reduce_sum(sq_err, axis=[1, 2]) / (64.0 * 64.0)

        # Apply importance weighting to wrists, ankles, and torso
        weighted_mse = mse_per_joint * self.weights_tensor

        # Normalize by total visible joints in the sample
        joint_vis = masks[:, 0, 0, :]
        total_vis = tf.reduce_sum(joint_vis * self.weights_tensor, axis=-1)

        per_sample_loss = tf.reduce_sum(weighted_mse, axis=-1) / tf.maximum(total_vis, 1.0)
        return tf.reduce_mean(per_sample_loss)
