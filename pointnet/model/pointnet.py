import tensorflow as tf
from .transformation import TNet
from .blocks import classification_net
from .layers import conv_block, dense_block


def PointNetClassifier(num_points, n_classes):
    input_tensor = tf.keras.Input(shape=(num_points, 3))
    x_t = TNet(input_tensor, num_points, 3)
    x = tf.matmul(input_tensor, x_t)
    x = conv_block(x, 32)
    x = conv_block(x, 32)
    x_t = TNet(x, num_points, 32)
    x = tf.matmul(x, x_t)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 512)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    output_tensor = classification_net(x, n_classes)
    return tf.keras.Model(input_tensor, output_tensor)
