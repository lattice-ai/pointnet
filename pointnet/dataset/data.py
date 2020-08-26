import tensorflow as tf


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


def get_dataset(tfrecord_files):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(
        tfrecord_files,
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    ).with_options(ignore_order).map(read_labeled_tfrecord)
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(
        tf.data.experimental.AUTOTUNE
    )
    return dataset
