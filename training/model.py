import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INPUT_HEIGHT = 256
INPUT_WIDTH = 256
INPUT_CHANNELS = 3
NUM_JOINTS = 16

def build_pose_model():
    inputs = keras.Input((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), name="input_image")
    x = layers.Conv2D(32, 3, strides=2, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", name="conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding="same", name="conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding="same", name="conv4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    outputs = layers.Conv2D(NUM_JOINTS, 1, padding="same",
                            activation=None, name="joint_heatmaps")(x)
    return keras.Model(inputs, outputs, name="FitnessVision_Pose_CNN")

if __name__ == "__main__":
    model = build_pose_model()
    model.summary()
    output = model(tf.random.normal((1, 256, 256, 3)))
    print("Output shape:", output.shape)
    assert output.shape == (1, 64, 64, 16)
