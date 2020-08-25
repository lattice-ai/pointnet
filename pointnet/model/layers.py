import tensorflow as tf


def conv_block(input_tensor, filters):
    x = tf.keras.layers.Conv1D(filters, kernel_size=1)(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


def dense_block(input_tensor, units):
    x = tf.keras.layers.Dense(units)(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)