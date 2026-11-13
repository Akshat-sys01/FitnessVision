import tensorflow as tf

def masked_mse(y_true, y_pred, joint_mask):
    """Masked heatmap MSE; joint_mask is (batch, 16), 1=visible, 0=excluded."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    joint_mask = tf.cast(joint_mask, tf.float32)
    per_joint = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2])
    masked = per_joint * joint_mask
    visible_count = tf.reduce_sum(joint_mask)
    return tf.reduce_sum(masked) / tf.maximum(visible_count, 1.0)

if __name__ == "__main__":
    truth = tf.ones((2, 64, 64, 16))
    pred = tf.zeros((2, 64, 64, 16))
    mask = tf.ones((2, 16))
    value = masked_mse(truth, pred, mask)
    print("Test loss:", float(value.numpy()))
    assert abs(float(value.numpy()) - 1.0) < 1e-6
    print("Loss validation passed.")
