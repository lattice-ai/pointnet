#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for managing modelnet10 dataset settings"""

from pydantic import BaseSettings


# !pylint:disable=too-few-public-methods
class Settings(BaseSettings):
    """Settings for dataset management.

    Pydantic entity containing every information required for obtaining
    dataset with tf.keras.util.get_file(), and additional configuration
    pertaining to number of classes and number of points to be sampled in each
    mesh.

    Args:
        origin:
            Original URL of the file.
        fname:
            Name of the file. If an absolute path /path/to/file.txt is
            specified the file will be saved at that location.
        cache_subdir:
            Subdirectory under the Keras cache dir where the file is saved. If
            an absolute path /path/to/folder is specified the file will be
            saved at that location.
        cache_dir:
            Location to store cached files, when None it defaults to the
            default directory ~/.keras/.
        extract:
            True tries extracting the file as an Archive, like tar or zip.
        num_classes:
            number of classes in the dataset.
        mesh_cardinality:
            number of points to be sampled for each mesh
    """
    origin: str = \
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/" \
        "ModelNet10.zip"
    fname: str = "ModelNet10.zip"
    cache_subdir: str = "modelnet10"
    cache_dir: str = "./datasets"
    extract: bool = True
    num_classes: int = 10
    mesh_cardinality: int = 2048
    batch_size: int = 8
    shuffle_buffer_size: int = 1000
    subfolder_glob: str = "[!README]*"
    file_glob_fmt: str = "{subset}/*"

    class Config:
        """Nested Config class for specifying file to load settings from."""

        env_file = "dataset.env"
