import math
import tensorflow as tf
from tensorflow.python import data
from tensorflow.python.keras.backend import dtype


def read_labeled_tfrecord(example):
    example = tf.io.parse_single_example(
        example, {
            'x': tf.io.FixedLenFeature([2048], tf.float32),
            'y': tf.io.FixedLenFeature([2048], tf.float32),
            'z': tf.io.FixedLenFeature([2048], tf.float32),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
    )
    mesh = tf.stack(
        [example['x'], example['y'], example['z']], axis=-1)
    label = example['label']
    return mesh, label


def apply_jitter(mesh, label):
    mesh += tf.random.normal(
        mesh.shape, mean=0.0, stddev=0.02,
        dtype=tf.float32, seed=69
    )
    return mesh, label


def apply_rotation(mesh, label):
    # rotated_data = tf.zeros(mesh.shape, dtype=tf.float32)
    theta = tf.random.uniform([]) * 2 * tf.constant(math.pi)
    sin_theta = tf.math.sin(theta)
    cos_theta = tf.math.cos(theta)
    rotation_matrix = tf.convert_to_tensor(
        [[cos_theta, 0, sin_theta],
        [0, 1, 0], [-sin_theta, 0, cos_theta]]
    )
    rotated_mesh = tf.tensordot(mesh, rotation_matrix)
    return rotated_mesh, label


def get_dataset(tfrecord_files, buffer_size, batch_size, augment):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(
        tfrecord_files,
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    ).with_options(ignore_order).map(read_labeled_tfrecord)
    if augment:
        dataset = dataset.map(apply_rotation)
        dataset = dataset.map(apply_jitter)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(
        tf.data.experimental.AUTOTUNE
    )
    return dataset
