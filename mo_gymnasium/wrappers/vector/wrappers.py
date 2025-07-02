"""Vector wrappers."""

import multiprocessing
import sys
import time
import traceback
from copy import deepcopy
from multiprocessing import Array, Queue
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, Iterator, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.error import NoAsyncCallError
from gymnasium.spaces.utils import is_space_dtype_shape_equiv
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.vector.async_vector_env import AsyncState
from gymnasium.vector.utils import (
    concatenate,
    create_empty_array,
    iterate,
    write_to_shared_memory,
)
from gymnasium.vector.vector_env import ArrayType, AutoresetMode, VectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics


class MOSyncVectorEnv(SyncVectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Example:
        >>> import mo_gymnasium as mo_gym

        >>> envs = mo_gym.wrappers.vector.MOSyncVectorEnv([
        ...     lambda: mo_gym.make("deep-sea-treasure-v0") for _ in range(4)
        ... ])
        >>> envs
        MOSyncVectorEnv(num_envs=4)
        >>> obs, infos = envs.reset()
        >>> obs
        array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=int32)
        >>> _ = envs.action_space.seed(42)
        >>> actions = envs.action_space.sample()
        >>> obs, rewards, terminateds, truncateds, infos = envs.step([0, 1, 2, 3])
        >>> obs
        array([[0, 0], [1, 0], [0, 0], [0, 3]], dtype=int32)
        >>> rewards
        array([[0., -1.], [0.7, -1.], [0., -1.], [0., -1.]], dtype=float32)
        >>> terminateds
        array([False,  True, False, False])
    """

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
        SyncVectorEnv.__init__(self, env_fns, copy=copy)
        # Just overrides the rewards memory to add the number of objectives
        self.reward_space = self.envs[0].unwrapped.reward_space
        self._rewards = np.zeros(
            (
                self.num_envs,
                self.reward_space.shape[0],
            ),
            dtype=np.float32,
        )

    def step(self, actions: ActType) -> Tuple[ObsType, ArrayType, ArrayType, ArrayType, Dict[str, Any]]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        actions = iterate(self.action_space, actions)

        observations, infos = [], {}
        for i, action in enumerate(actions):
            if self._autoreset_envs[i]:
                env_obs, env_info = self.envs[i].reset()

                self._rewards[i] = np.zeros(self.reward_space.shape[0])  # This overrides Gymnasium's implem
                self._terminations[i] = False
                self._truncations[i] = False
            else:
                (
                    env_obs,
                    self._rewards[i],
                    self._terminations[i],
                    self._truncations[i],
                    env_info,
                ) = self.envs[
                    i
                ].step(action)

            observations.append(env_obs)
            infos = self._add_info(infos, env_info, i)

        # Concatenate the observations
        self._observations = concatenate(self.single_observation_space, observations, self._observations)
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            deepcopy(self._observations) if self.copy else self._observations,
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )


def _mo_async_worker(
    index: int,
    env_fn: callable,
    pipe: Connection,
    parent_pipe: Connection,
    shared_memory: Union[Array, Dict[str, Any], Tuple[Any, ...]],
    error_queue: Queue,
    autoreset_mode: AutoresetMode,
):
    env = env_fn()
    observation_space = env.observation_space
    action_space = env.action_space
    reward_space = env.unwrapped.reward_space
    autoreset = False
    observation = None

    parent_pipe.close()

    try:
        while True:
            command, data = pipe.recv()

            if command == "reset":
                observation, info = env.reset(**data)
                if shared_memory:
                    write_to_shared_memory(observation_space, index, observation, shared_memory)
                    observation = None
                    autoreset = False
                pipe.send(((observation, info), True))
            elif command == "reset-noop":
                pipe.send(((observation, {}), True))
            elif command == "step":
                if autoreset_mode == AutoresetMode.NEXT_STEP:
                    if autoreset:
                        observation, info = env.reset()
                        reward, terminated, truncated = (
                            np.zeros(reward_space.shape[0], dtype=np.float32),
                            False,
                            False,
                        )
                    else:
                        (
                            observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = env.step(data)
                    autoreset = terminated or truncated
                elif autoreset_mode == AutoresetMode.SAME_STEP:
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(data)

                    if terminated or truncated:
                        reset_observation, reset_info = env.reset()

                        info = {
                            "final_info": info,
                            "final_obs": observation,
                            **reset_info,
                        }
                        observation = reset_observation
                elif autoreset_mode == AutoresetMode.DISABLED:
                    assert autoreset is False
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(data)
                else:
                    raise ValueError(f"Unexpected autoreset_mode: {autoreset_mode}")

                if shared_memory:
                    write_to_shared_memory(observation_space, index, observation, shared_memory)
                    observation = None

                pipe.send(((observation, reward, terminated, truncated, info), True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "close", "_setattr", "_check_spaces"]:
                    raise ValueError(f"Trying to call function `{name}` with `call`, use `{name}` directly instead.")

                attr = env.get_wrapper_attr(name)
                if callable(attr):
                    pipe.send((attr(*args, **kwargs), True))
                else:
                    pipe.send((attr, True))
            elif command == "_setattr":
                name, value = data
                env.set_wrapper_attr(name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                obs_mode, single_obs_space, single_action_space = data

                pipe.send(
                    (
                        (
                            (
                                single_obs_space == observation_space
                                if obs_mode == "same"
                                else is_space_dtype_shape_equiv(single_obs_space, observation_space)
                            ),
                            single_action_space == action_space,
                        ),
                        True,
                    )
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must be one of [`reset`, `step`, `close`, `_call`, `_setattr`, `_check_spaces`]."
                )
    except (KeyboardInterrupt, Exception):
        error_type, error_message, _ = sys.exc_info()
        trace = traceback.format_exc()

        error_queue.put((index, error_type, error_message, trace))
        pipe.send((None, False))
    finally:
        env.close()


class MOAsyncVectorEnv(AsyncVectorEnv):
    """Vectorized environment that runs multiple environments in parallel.

    It uses ``multiprocessing`` processes, and pipes for communication.

    Modified from gymnasium.vector.async_vector_env.AsyncVectorEnv to allow for multi-objective rewards.

    Example:
        >>> import mo_gymnasium as mo_gym
        >>> envs = mo_gym.wrappers.vector.MOAsyncVectorEnv([
        ...     lambda: mo_gym.make("deep-sea-treasure-v0") for _ in range(4)
        ... ])
        >>> envs
        MOAsyncVectorEnv(num_envs=4)
        >>> obs, infos = envs.reset()
        >>> obs
        array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=int32)
        >>> _ = envs.action_space.seed(42)
        >>> actions = envs.action_space.sample()
        >>> obs, rewards, terminateds, truncateds, infos = envs.step([0, 1, 2, 3])
        >>> obs
        array([[0, 0], [1, 0], [0, 0], [0, 3]], dtype=int32)
        >>> rewards
        array([[0., -1.], [0.7, -1.], [0., -1.], [0., -1.]], dtype=float32)
        >>> terminateds
        array([False,  True, False, False])
    """

    def __init__(self, env_fns: Sequence[Callable[[], gym.Env]], **kwargs):
        """Vectorized environment that runs multiple environments in parallel.

        Args:
            env_fns: env constructors
        """
        super().__init__(env_fns=env_fns, worker=_mo_async_worker, **kwargs)

        # extract reward space from first vector env and create 2d array to store vector rewards
        dummy_env = env_fns[0]()
        self.reward_space = dummy_env.unwrapped.reward_space
        dummy_env.close()
        del dummy_env
        self.rewards = create_empty_array(self.reward_space, n=self.num_envs, fn=np.zeros)

    def step_wait(self, timeout: int | float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Wait for the calls to :obj:`step` in each sub-environment to finish.

        Args:
            timeout: Number of seconds before the call to :meth:`step_wait` times out. If ``None``, the call to :meth:`step_wait` never times out.

        Returns:
             The batched environment step information, (obs, reward, terminated, truncated, info)

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            NoAsyncCallError: If :meth:`step_wait` was called without any prior call to :meth:`step_async`.
            TimeoutError: If :meth:`step_wait` timed out.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll_pipe_envs(timeout):
            self._state = AsyncState.DEFAULT
            raise multiprocessing.TimeoutError(f"The call to `step_wait` has timed out after {timeout} second(s).")

        observations, rewards, terminations, truncations, infos = [], [], [], [], {}
        successes = []
        for env_idx, pipe in enumerate(self.parent_pipes):
            env_step_return, success = pipe.recv()

            successes.append(success)
            if success:
                observations.append(env_step_return[0])
                rewards.append(env_step_return[1])
                terminations.append(env_step_return[2])
                truncations.append(env_step_return[3])
                infos = self._add_info(infos, env_step_return[4], env_idx)

        self._raise_if_errors(successes)

        if not self.shared_memory:
            self.observations = concatenate(
                self.single_observation_space,
                observations,
                self.observations,
            )

        # modify to allow return of vector rewards
        self.rewards = concatenate(
            self.reward_space,
            rewards,
            self.rewards,
        )

        self._state = AsyncState.DEFAULT
        return (
            deepcopy(self.observations) if self.copy else self.observations,
            deepcopy(self.rewards) if self.copy else self.rewards,
            np.array(terminations, dtype=np.bool_),
            np.array(truncations, dtype=np.bool_),
            infos,
        )


class MORecordEpisodeStatistics(RecordEpisodeStatistics):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of any episode within the vectorized env, the statistics of the episode
    will be added to ``info`` using the key ``episode``, and the ``_episode`` key
    is used to indicate the environment index which has a terminated or truncated episode.

     For a vectorized environments the output will be in the form of (be careful to first wrap the env into vector before applying MORewordStatistics)::

        >>> infos = { # doctest: +SKIP
        ...     "episode": {
        ...         "r": "<array of cumulative reward for each done sub-environment (2d array, shape (num_envs, dim_reward))>",
        ...         "dr": "<array of discounted reward for each done sub-environment (2d array, shape (num_envs, dim_reward))>",
        ...         "l": "<array of episode length for each done sub-environment (array)>",
        ...         "t": "<array of elapsed time since beginning of episode for each done sub-environment (array)>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(
        self,
        env: VectorEnv,
        gamma: float = 1.0,
        buffer_length: int = 100,
        stats_key: str = "episode",
    ):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            gamma: The discount factor
            buffer_length: The size of the buffers :attr:`return_queue`, :attr:`length_queue` and :attr:`time_queue`
            stats_key: The info key to save the data
        """
        gym.utils.RecordConstructorArgs.__init__(self, buffer_length=buffer_length, stats_key=stats_key)
        RecordEpisodeStatistics.__init__(self, env, buffer_length=buffer_length, stats_key=stats_key)
        self.disc_episode_returns = None
        self.reward_dim = self.env.unwrapped.reward_space.shape[0]
        self.rewards_shape = (self.num_envs, self.reward_dim)
        self.gamma = gamma

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)

        # CHANGE: Here we just override the standard implementation to extend to MO
        self.episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)
        self.disc_episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)

        return obs, info

    def step(self, actions: ActType) -> Tuple[ObsType, ArrayType, ArrayType, ArrayType, Dict[str, Any]]:
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)

        assert isinstance(
            infos, dict
        ), f"`vector.RecordEpisodeStatistics` requires `info` type to be `dict`, its actual type is {type(infos)}. This may be due to usage of other wrappers in the wrong order."

        self.episode_returns[self.prev_dones] = 0
        self.episode_lengths[self.prev_dones] = 0
        self.episode_start_times[self.prev_dones] = time.perf_counter()
        self.episode_returns[~self.prev_dones] += rewards[~self.prev_dones]

        # CHANGE: The discounted returns are also computed here
        self.disc_episode_returns += rewards * np.repeat(self.gamma**self.episode_lengths, self.reward_dim).reshape(
            self.episode_returns.shape
        )
        self.episode_lengths[~self.prev_dones] += 1

        self.prev_dones = dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if self._stats_key in infos or f"_{self._stats_key}" in infos:
                raise ValueError(f"Attempted to add episode stats when they already exist, info keys: {list(infos.keys())}")
            else:
                # CHANGE to handle the vectorial reward and do deepcopies
                episode_return = np.zeros(self.rewards_shape, dtype=np.float32)
                disc_episode_return = np.zeros(self.rewards_shape, dtype=np.float32)

                for i in range(self.num_envs):
                    if dones[i]:
                        episode_return[i] = np.copy(self.episode_returns[i])
                        disc_episode_return[i] = np.copy(self.disc_episode_returns[i])

                episode_time_length = np.round(time.perf_counter() - self.episode_start_times, 6)
                infos[self._stats_key] = {
                    "r": episode_return,
                    "dr": disc_episode_return,
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(dones, episode_time_length, 0.0),
                }
                infos[f"_{self._stats_key}"] = dones

            self.episode_count += num_dones

            for i in np.where(dones):
                self.time_queue.extend(episode_time_length[i])
                self.return_queue.extend(self.episode_returns[i])
                self.length_queue.extend(self.episode_lengths[i])

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
