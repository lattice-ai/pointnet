#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module to provide model settings."""

from pydantic import BaseSettings


# !pylint:disable=too-few-public-methods
class Settings(BaseSettings):
    """Settings class for model configuration.

    Args:
        learning_rate:
            learning rate for optimizer. This setting is specified as:
                static:<floating point learning rate>
                | dynamic:<string representing learning rate scheduler>
    """

    learning_rate: str = "static:1e-3"

    class Config:
        """Nested Config class for specifying file to read settings from."""
        env_file = "model.env"
