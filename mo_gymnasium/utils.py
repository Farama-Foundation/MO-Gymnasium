"""Utilities function such as wrappers."""

import time
from copy import deepcopy
from typing import Iterator, Tuple, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.utils import EzPickle
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers.normalize import RunningMeanStd


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


class LinearReward(gym.Wrapper, EzPickle):
    """Makes the env return a scalar reward, which is the dot-product between the reward vector and the weight vector."""

    def __init__(self, env: gym.Env, weight: np.ndarray = None):
        """Makes the env return a scalar reward, which is the dot-product between the reward vector and the weight vector.

        Args:
            env: env to wrap
            weight: weight vector to use in the dot product
        """
        super().__init__(env)
        EzPickle.__init__(self, env, weight)
        if weight is None:
            weight = np.ones(shape=env.reward_space.shape)
        self.set_weight(weight)

    def set_weight(self, weight: np.ndarray):
        """Changes weights for the scalarization.

        Args:
            weight: new weights to set
        Returns: nothing
        """
        assert weight.shape == self.env.reward_space.shape, "Reward weight has different shape than reward vector."
        self.w = weight

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Steps in the environment.

        Args:
            action: action to perform
        Returns: obs, scalarized_reward, terminated, truncated, info
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        scalar_reward = np.dot(reward, self.w)
        info["vector_reward"] = reward

        return observation, scalar_reward, terminated, truncated, info


class MONormalizeReward(gym.Wrapper, EzPickle):
    """Wrapper to normalize the reward component at index idx. Does not touch other reward components."""

    def __init__(self, env: gym.Env, idx: int, gamma: float = 0.99, epsilon: float = 1e-8):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            idx (int): the index of the reward to normalize
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        EzPickle.__init__(self, env, idx, gamma, epsilon)
        self.idx = idx
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action: ActType):
        """Steps through the environment, normalizing the rewards returned.

        Args:
            action: action to perform
        Returns: obs, normalized_rewards, terminated, truncated, infos
        """
        obs, rews, terminated, truncated, infos = self.env.step(action)
        # Extracts the objective value to normalize
        to_normalize = rews[self.idx]
        if not self.is_vector_env:
            to_normalize = np.array([to_normalize])
        self.returns = self.returns * self.gamma + to_normalize
        # Defer normalization to gym implementation
        to_normalize = self.normalize(to_normalize)
        self.returns[terminated] = 0.0
        if not self.is_vector_env:
            to_normalize = to_normalize[0]
        # Injecting the normalized objective value back into the reward vector
        rews[self.idx] = to_normalize
        return obs, rews, terminated, truncated, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance.

        Args:
            rews: rewards
        Returns: the normalized reward
        """
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)


class MOClipReward(gym.RewardWrapper, EzPickle):
    """Clip reward[idx] to [min, max]."""

    def __init__(self, env: gym.Env, idx: int, min_r, max_r):
        """Clip reward[idx] to [min, max].

        Args:
            env: environment to wrap
            idx: index of the MO reward to clip
            min_r: min reward
            max_r: max reward
        """
        super().__init__(env)
        EzPickle.__init__(self, env, idx, min_r, max_r)
        self.idx = idx
        self.min_r = min_r
        self.max_r = max_r

    def reward(self, reward):
        """Clips the reward at the given index.

        Args:
            reward: reward to clip.
        Returns: the clipped reward.
        """
        reward[self.idx] = np.clip(reward[self.idx], self.min_r, self.max_r)
        return reward


class MOSyncVectorEnv(SyncVectorEnv, EzPickle):
    """Vectorized environment that serially runs multiple environments."""

    def __init__(
        self,
        env_fns: Iterator[callable],
        copy: bool = True,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: env constructors
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.
        """
        super().__init__(env_fns, copy=copy)
        EzPickle.__init__(self, env_fns, copy=copy)
        # Just overrides the rewards memory to add the number of objectives
        self.reward_space = self.envs[0].reward_space
        self._rewards = np.zeros(
            (
                self.num_envs,
                self.reward_space.shape[0],
            ),
            dtype=np.float64,
        )


class MORecordEpisodeStatistics(RecordEpisodeStatistics, EzPickle):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward (array)>",
        ...         "dr": "<discounted reward (array)>",
        ...         "l": "<episode length (scalar)>", # contrary to Gymnasium, these are not a numpy array
        ...         "t": "<elapsed time since beginning of episode (scalar)>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward (2d array, shape (num_envs, dim_reward))>",
        ...         "dr": "<array of discounted reward (2d array, shape (num_envs, dim_reward))>",
        ...         "l": "<array of episode length (array)>",
        ...         "t": "<array of elapsed time since beginning of episode (array)>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }
    """

    def __init__(self, env: gym.Env, gamma: float = 1.0, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            gamma (float): Discounting factor
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        super().__init__(env, deque_size)
        EzPickle.__init__(self, env, gamma, deque_size)
        # CHANGE: Here we just override the standard implementation to extend to MO
        # We also take care of the case where the env is vectorized
        self.reward_dim = self.env.reward_space.shape[0]
        if self.is_vector_env:
            self.rewards_shape = (self.num_envs, self.reward_dim)
        else:
            self.rewards_shape = (self.reward_dim,)
        self.gamma = gamma

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)

        # CHANGE: Here we just override the standard implementation to extend to MO
        self.episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)
        self.disc_episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)

        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        # This is very close the code from the RecordEpisodeStatistics wrapper from gym.
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1

        # CHANGE: The discounted returns are also computed here
        self.disc_episode_returns += rewards * np.repeat(self.gamma**self.episode_lengths, self.reward_dim).reshape(
            self.episode_returns.shape
        )

        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError("Attempted to add episode stats when they already exist")
            else:
                episode_return = np.zeros(self.rewards_shape, dtype=np.float32)
                disc_episode_return = np.zeros(self.rewards_shape, dtype=np.float32)
                if self.is_vector_env:
                    for i in range(self.num_envs):
                        if dones[i]:
                            # CHANGE: Makes a deepcopy to avoid subsequent mutations
                            episode_return[i] = deepcopy(self.episode_returns[i])
                            disc_episode_return[i] = deepcopy(self.disc_episode_returns[i])
                else:
                    episode_return = deepcopy(self.episode_returns)
                    disc_episode_return = deepcopy(self.disc_episode_returns)

                length_eps = np.where(dones, self.episode_lengths, 0)
                time_eps = np.where(
                    dones,
                    np.round(time.perf_counter() - self.episode_start_times, 6),
                    0.0,
                )

                infos["episode"] = {
                    "r": episode_return,
                    "dr": disc_episode_return,
                    "l": length_eps[0] if not self.is_vector_env else length_eps,
                    "t": time_eps[0] if not self.is_vector_env else time_eps,
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = np.zeros(self.reward_dim, dtype=np.float32)
            self.disc_episode_returns[dones] = np.zeros(self.reward_dim, dtype=np.float32)
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
