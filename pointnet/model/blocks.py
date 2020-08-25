import tensorflow as tf
from .layers import dense_block


def classification_net(input_tensor, n_classes):
    x = dense_block(input_tensor, 512)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = dense_block(x, 256)
    x = tf.keras.layers.Dropout(0.3)(x)
    return tf.keras.layers.Dense(
        n_classes, activation="softmax"
    )(x)
