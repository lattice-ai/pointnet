import tensorflow as tf


class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, features, l2) -> None:
        super(OrthogonalRegularizer, self).__init__()
        self.features = features
        self.l2 = l2
        self.I = tf.eye(features)
    
    def call(self, inputs):
        A = tf.reshape(inputs, (-1, self.features, self.features))
        AAT = tf.tensordot(A, A, axes=(2, 2))
        AAT = tf.reshape(AAT, (-1, self.features, self.features))
        l_reg = tf.reduce_sum(self.l2 * tf.square(AAT - self.I))
        return l_reg
