"""
Custom Loss Functions and Evaluation Metrics for Pose Estimation.
- MaskedHuberLoss: Smooth L1 loss applied strictly to annotated/visible joints.
- MaskedMSELoss: Mean squared error applied strictly to annotated/visible joints.
- PCKMetric: Percentage of Correct Keypoints within threshold distance.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="PoseEstimation")
def masked_huber_loss(y_true, y_pred, delta=0.05):
    target_kps = y_true[:, :32]
    target_mask = y_true[:, 32:]

    error = tf.abs(target_kps - y_pred)
    is_small = error <= delta
    huber = tf.where(is_small, 0.5 * tf.square(error), delta * (error - 0.5 * delta))

    masked_loss = tf.reduce_sum(huber * target_mask) / (tf.reduce_sum(target_mask) + 1e-6)

    # Spread penalty: penalize if all predicted joints collapse to same point
    pred_xy = tf.reshape(y_pred, [-1, 16, 2])
    pred_mean = tf.reduce_mean(pred_xy, axis=1, keepdims=True)
    spread = tf.reduce_mean(tf.reduce_sum(tf.square(pred_xy - pred_mean), axis=-1))
    spread_penalty = tf.maximum(0.0, 0.01 - spread) * 2.0  # penalize collapse

    return masked_loss + spread_penalty


@tf.keras.utils.register_keras_serializable(package="PoseEstimation")
def masked_mse_loss(y_true, y_pred):
    """
    Computes MSE loss ignoring unannotated joints.
    """
    target_kps = y_true[:, :32]
    target_mask = y_true[:, 32:]
    
    sq_err = tf.square(target_kps - y_pred)
    masked_err = sq_err * target_mask
    total_mask = tf.reduce_sum(target_mask) + 1e-6
    return tf.reduce_sum(masked_err) / total_mask


@tf.keras.utils.register_keras_serializable(package="PoseEstimation")
class PCKMetric(tf.keras.metrics.Metric):
    """
    Percentage of Correct Keypoints (PCK@threshold).
    Measures accuracy of predicted 2D joint coordinates.
    Threshold default = 0.08 (approx 18px on 224x224 image).
    """
    def __init__(self, threshold=0.08, name="pck_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct_joints = self.add_weight(name="correct", initializer="zeros")
        self.total_joints = self.add_weight(name="total", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        target_kps = y_true[:, :32]
        target_mask = y_true[:, 32:]
        
        # Reshape to (batch, 16, 2)
        true_xy = tf.reshape(target_kps, [-1, 16, 2])
        pred_xy = tf.reshape(y_pred, [-1, 16, 2])
        mask_xy = tf.reshape(target_mask, [-1, 16, 2])[:, :, 0] # (batch, 16)
        
        # Euclidean distance per joint
        dist = tf.norm(true_xy - pred_xy, axis=-1) # (batch, 16)
        
        is_correct = tf.cast(dist <= self.threshold, tf.float32) * mask_xy
        
        self.correct_joints.assign_add(tf.reduce_sum(is_correct))
        self.total_joints.assign_add(tf.reduce_sum(mask_xy))
        
    def result(self):
        return tf.math.divide_no_nan(self.correct_joints, self.total_joints)
        
    def reset_state(self):
        self.correct_joints.assign(0.0)
        self.total_joints.assign(0.0)
        
    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config
