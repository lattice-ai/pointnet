import numpy as np
import tensorflow as tf
from .layers import conv_block, dense_block
from .regularizers import OrthogonalRegularizer


def TNet(input_tensor, num_points, features):
    x = conv_block(input_tensor, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 1024)
    # x = tf.keras.layers.MaxPooling1D(pool_size=num_points)(x)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = dense_block(x, 512)
    x = dense_block(x, 256)
    x = tf.keras.layers.Dense(
        features * features,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(
            tf.reshape(tf.cast(tf.eye(features), dtype=tf.float32), [-1])
        ),
        activity_regularizer=OrthogonalRegularizer(features)
    )(x)
    x = tf.reshape(x, (-1, features, features))
    return x
