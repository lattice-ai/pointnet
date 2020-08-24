import os
import tensorflow as tf


def download_dataset(url: str, filename: str, dataset_name: str):
    data_path = tf.keras.utils.get_file(
        filename, url, extract=True
    )
    data_path = os.path.join(
        os.path.dirname(data_path), dataset_name)
    return data_path
