import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters) -> None:
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters, kernel_size=1, padding='valid')
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=0.0)
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)
        return x


class MLPBlock(tf.keras.layers.Layer):

    def __init__(self, units) -> None:
        super(ConvBlock, self).__init__()
        self.dense = tf.keras.layers.Dense(units)
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=0.0)
    
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)
        return x
