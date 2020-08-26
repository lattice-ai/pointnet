import os
from pointnet.dataset import TFRecordCreator, download_dataset


def generate_tfrecords():
    data_path = download_dataset(
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        "modelnet10.zip", "ModelNet10"
    )
    recorder_train = TFRecordCreator(data_path, 'train', 1000)
    recorder_test = TFRecordCreator(data_path, 'test', 1000)
    try: os.mkdir('./tfrecords'); os.mkdir('./tfrecords/train'); os.mkdir('./tfrecords/test')
    except: pass
    recorder_train.create_records('./tfrecords')
    recorder_test.create_records('./tfrecords')


if __name__ == '__main__':
    generate_tfrecords()
