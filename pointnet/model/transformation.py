import tensorflow as tf
from .layers import ConvBlock, MLPBlock
from .regularizers import OrthogonalRegularizer


class TNet(tf.keras.layers.Layer):

    def __init__(self, features) -> None:
        super(TNet, self).__init__()
        self.conv_blocks = [ConvBlock(32), ConvBlock(64), ConvBlock(512)]
        self.max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.mlp_blocks = [MLPBlock(256), MLPBlock(128)]
        self.output_dense = tf.keras.layers.Dense(
            features * features, kernel_initializer='zeros',
            bias_initializer=tf.keras.initializers.Constant(
                tf.reshape(tf.eye(5), [-1])
            ), activity_regularizer=OrthogonalRegularizer(features)
        )
        self.reshape = tf.keras.layers.Reshape((features, features))
        self.matrix_multiply = tf.keras.layers.Dot(axes=(2, 1))
    
    def call(self, inputs):
        x = self.conv_blocks[0](inputs)
        x = self.conv_blocks[1](x)
        x = self.conv_blocks[2](x)
        x = self.max_pool(x)
        x = self.mlp_blocks[0](x)
        x = self.mlp_blocks[1](x)
        x = self.output_dense(x)
        x = self.reshape(x)
        x = self.matrix_multiply([inputs, x])
        return x
