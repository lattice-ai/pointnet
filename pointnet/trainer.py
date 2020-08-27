import os
import wandb
import tensorflow as tf
from .model import PointNetClassifier
from .dataset.data import get_dataset


class Trainer:

    def __init__(self, configs) -> None:
        self.configs = configs
        os.environ['WANDB_API_KEY'] = self.configs['wandb-api-key']
        wandb.init(project=self.configs['project'], name=self.configs['experiment-name'])
        self.train_dataset = get_dataset(
            self.configs['train_tfrecord_files'], self.configs['buffer_size'],
            batch_size=self.configs['batch_size'], augment=True
        )
        self.val_dataset = get_dataset(
            self.configs['test_tfrecord_files'], self.configs['buffer_size'],
            batch_size=self.configs['batch_size'], augment=False
        )
        self.model = PointNetClassifier(
            self.configs['num_points'],
            self.configs['n_classes']
        )
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.configs['learning-rate']),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
    
    def train(self):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    wandb.run.dir,
                    self.configs['weight_file']
                ),
                monitor='val_loss', save_best_only=True,
                mode='min', save_weights_only=True
            ), wandb.keras.WandbCallback()
        ]
        history = self.model.fit(
            self.train_dataset, epochs=self.configs['epochs'],
            validation_data=self.val_dataset, callbacks=callbacks
        )
        return history
