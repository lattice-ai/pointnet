from glob import glob
from pointnet.trainer import Trainer, tf


def run_training():
    configs = {
        'wandb-api-key': '4c77a6750a931c1b13d4d10a0e058725a7487ba9',
        'experiment-name': 'classification-modelnet10', 'project': 'pointnet',
        'train_tfrecord_files': glob('./tfrecords/train*'), 'train_files': 4000,
        'test_tfrecord_files': glob('./tfrecords/test*'), 'test_files': 908,
        'buffer_size': 2048, 'batch_size': 32, 'num_points': 2048, 'n_classes': 10,
        'weight_file': 'classification-modelnet10', 'epochs': 50,
        'learning-rate': tf.keras.optimizers.schedules.PolynomialDecay(
            0.001, 10000, end_learning_rate=0.0001,
            power=0.9, cycle=False, name=None
        )
    }

    trainer = Trainer(configs)
    trainer.train()


if __name__ == '__main__':
    run_training()
