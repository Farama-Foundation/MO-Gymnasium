"""Utilities functions."""

from typing import TypeVar

import gymnasium as gym


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def make(env_name: str, disable_env_checker: bool = True, **kwargs) -> gym.Env:
    """Overrides Gymnasium's make method to disable env_checker by default.

    Args:
        env_name: name of the environment to create
        disable_env_checker: disables environment checker
        **kwargs: forwards arguments to the environment constructor
    Returns: a newly created environment.
    """
    """Disable env checker, as it requires the reward to be a scalar."""
    return gym.make(env_name, disable_env_checker=disable_env_checker, **kwargs)
