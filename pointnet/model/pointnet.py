import tensorflow as tf
from .transformation import TNet
from .layers import ConvBlock, MLPBlock
from .regularizers import OrthogonalRegularizer


class PointNetClassifier(tf.keras.Model):

    def __init__(self, n_classes) -> None:
        super(PointNetClassifier, self).__init__()
        self.transformations = [TNet(3), TNet(32)]
        self.conv_blocks = [
            ConvBlock(32), ConvBlock(32), ConvBlock(32),
            ConvBlock(32), ConvBlock(64), ConvBlock(512)
        ]
        self.max_pool = tf.keras.layers.GlobalMaxPooling1D()
        self.mlp_blocks = [MLPBlock(256), MLPBlock(128)]
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.output = tf.keras.layers.Dense(
            n_classes, activation="softmax"
        )
    
    def call(self, inputs):
        x = self.transformations[0](inputs)
        x = self.conv_blocks[0](x)
        x = self.conv_blocks[1](x)
        x = self.transformations[1](inputs)
        x = self.conv_blocks[2](x)
        x = self.conv_blocks[3](x)
        x = self.conv_blocks[4](x)
        x = self.max_pool(x)
        x = self.mlp_blocks[0](x)
        x = self.dropout(x)
        x = self.mlp_blocks[1](x)
        x = self.dropout(x)
        x = self.output(x)
        return x
