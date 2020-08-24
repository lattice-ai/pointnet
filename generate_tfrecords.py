import os
from pointnet.dataset.tfrecord import TFRecordCreator
from pointnet.dataset.download import download_dataset


def generate_tfrecords():
    data_path = download_dataset(
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        "modelnet10.zip", "ModelNet10"
    )
    recorder = TFRecordCreator(data_path, 1000)
    try: os.mkdir('./tfrecords')
    except: pass
    recorder.create_records('./tfrecords')


if __name__ == '__main__':
    generate_tfrecords()
