#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for providing dataset manager."""

import os
from pathlib import Path
from glob import glob

from typing import Callable

import trimesh
import tensorflow as tf

import pointnet.config.dataset


class Manager:
    """Manager class for pointnet.dataset

    Args:
        config: dataset settings
    """
    def __init__(self, config: pointnet.config.dataset.Settings):
        self._config = config
        self._augmentations = []
        self._train_ds, self._test_ds = None, None
        self._data_dir = None
        self._class_map = None

    def fetch(self):
        """Fetch dataset using tf.keras.utils.get_file with the parameters
        available in the config."""
        if self._data_dir is not None:
            return self._data_dir

        os.makedirs(self._config.cache_dir, exist_ok=True)

        archive_path = tf.keras.utils.get_file(
            origin=self._config.origin,
            fname=self._config.fname,
            cache_subdir=self._config.cache_subdir,
            cache_dir=self._config.cache_dir,
            extract=self._config.extract
        )

        self._data_dir = Path(os.path.realpath(self._config.cache_dir),
                              self._config.cache_subdir,
                              Path(archive_path).stem)
        assert self._data_dir.is_dir()

        print("[+] Data dir: ", self._data_dir)

        self._data_dir = self._data_dir.__str__()
        return self._data_dir

    def register_augmentation(self, augmentation: Callable[[tf.Tensor, str],
                                                           tf.Tensor]):
        """Registers a new augmentation operation in the dataset pipeline.

        Args:
            augmentation: augmentation callable operation which maps a
            tf.Tensor to a tf.Tensor
        """
        self._augmentations.append(augmentation)

    @property
    def data_dir(self):
        """Property to get the data directory used by
        tf.keras.utils.get_file.

        Returns:
            str path to data directory.
        """
        return self.fetch()

    @property
    def class_map(self) -> dict:
        """Property to get the class to integer label mapping.

        Returns:
            dict mapping from class names to integer labels.
        """
        if self._class_map is None:
            subfolders = glob(
                Path(self.data_dir, self._config.subfolder_glob).__str__())

            self._class_map = {
                subfolder.split("/")[-1]: i
                for i, subfolder in enumerate(subfolders)
            }

        print("[+] Class Map: ", self._class_map)

        return self._class_map

    def _get_files_for_subset(self, subset):
        glob_fmt = Path(self.data_dir, self._config.subfolder_glob,
                        self._config.file_glob_fmt.format(
                            subset=subset
                        )).__str__()
        print("[-] Using glob fmt: ", glob_fmt)
        return glob(glob_fmt)

    def _get_label_from_mesh_path(self, path: str):
        return self.class_map[path.split("/")[-3]]

    def _read_mesh_with_label(self, path: tf.Tensor):
        path = str(path.numpy())

        assert Path(path).is_file()

        mesh = trimesh.load(path).sample(self._config.mesh_cardinality)
        label = self._get_label_from_mesh_path(path)
        return mesh, label

    @property
    def train_ds(self) -> tf.data.Dataset:
        """Property to access the train_ds.

        Returns:
            a tf.data.Dataset instance containing the training data as
            specified in the dataset settings config, with the registered
            augmentations.
        """
        if self._train_ds is not None:
            return self._train_ds

        train_files = self._get_files_for_subset(subset="train")

        print("Train files; train_files[:3] = ", train_files[:3])

        self._train_ds = tf.data.Dataset.from_tensor_slices(train_files)

        print("[+] Sample data point from train_ds before read op: ",
              next(iter(self._train_ds)))

        self._train_ds = self._train_ds.map(
            self._read_mesh_with_label).shuffle(buffer_size=len(train_files))

        for augmentation in self._augmentations:
            self._train_ds = self._train_ds.map(augmentation)

        self._train_ds = self._train_ds.batch(self._config.batch_size)

        return self._train_ds

    @property
    def test_ds(self) -> tf.data.Dataset:
        """Property to access the test_ds.

        Returns:
            a tf.data.Dataset instance containing the test data as parsed
            with #parse().
        """
        if self._test_ds is not None:
            return self._test_ds

        test_files = self._get_files_for_subset(subset="test")
        self._test_ds = tf.data.Dataset.from_tensor_slices(test_files)

        self._test_ds = self._test_ds

        self._test_ds = self._test_ds.map(
            self._read_mesh_with_label).shuffle(
                buffer_size=len(test_files)).batch(self._config.batch_size)

        return self._test_ds
