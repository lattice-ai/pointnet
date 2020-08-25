import tensorflow as tf


class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    
    def __init__(self, num_features, l2=0.001):
        self.num_features = num_features
        self.l2 = l2
        self.I = tf.eye(num_features)

    def __call__(self, inputs):
        A = tf.reshape(inputs, (-1, self.num_features, self.num_features))
        AAT = tf.tensordot(A, A, axes=(2, 2))
        AAT = tf.reshape(AAT, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2 * tf.square(AAT - self.I))
