import os
import pickle
import trimesh
from glob import glob
from tqdm import tqdm
import tensorflow as tf


class TFRecordCreator:

    def __init__(self, data_path, dataset_type, size) -> None:
        self.dataset_type = dataset_type
        self.model_files, self.classes, self.all_classes = self._get_files_and_classes(data_path)
        self.size = size
    
    def _get_files_and_classes(self, data_path):
        model_files = glob(os.path.join(data_path, '*/{}/*.off'.format(self.dataset_type)))
        classes = [model_file.split('/')[-3] for model_file in model_files]
        all_classes = list(set(classes))
        classes = [all_classes.index(label) for label in classes]
        return model_files, classes, all_classes
    
    def _create_float_feature(self, value):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=value)
        )
    
    def _create_int64_feature(self, value):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value])
        )
    
    def create_records(self, tfrecord_dir):
        print(); print('Dumping list of all classes...')
        with open('./tfrecords/classes.pkl', 'wb') as dump_file:
            pickle.dump(self.all_classes, dump_file)
        ct = len(self.model_files) // self.size + int(len(self.model_files) % self.size != 0)
        for j in range(ct):
            print(); print('Writing TFRecord %i of %i...'%(j + 1, ct))
            ct2 = min(self.size, len(self.model_files) - j * self.size)
            with tf.io.TFRecordWriter(
                os.path.join(
                    tfrecord_dir,
                    self.dataset_type + '%.2i-%i.tfrec' % (j + 1, ct2))
            ) as writer:
                for k in tqdm(range(ct2)):
                    x, y, z = trimesh.load(
                        self.model_files[self.size * j + k]
                    ).sample(2048).T
                    label = self.classes[self.size * j + k]
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'x': self._create_float_feature(x.flatten()),
                                'y': self._create_float_feature(y.flatten()),
                                'z': self._create_float_feature(z.flatten()),
                                'label': self._create_int64_feature(label)
                            }
                        )
                    )
                    writer.write(example.SerializeToString())
