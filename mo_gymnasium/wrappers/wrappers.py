"""Wrappers."""

import os
import time
import imageio
from copy import deepcopy
from typing import Tuple, TypeVar, Callable

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.common import RecordEpisodeStatistics
from gymnasium.wrappers.utils import RunningMeanStd
from gymnasium.utils.save_video import capped_cubic_video_schedule

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LinearReward(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Makes the env return a scalar reward, which is the dot-product between the reward vector and the weight vector."""

    def __init__(self, env: gym.Env, weight: np.ndarray = None):
        """Makes the env return a scalar reward, which is the dot-product between the reward vector and the weight vector.

        Args:
            env: env to wrap
            weight: weight vector to use in the dot product
        """
        gym.utils.RecordConstructorArgs.__init__(self, weight=weight)
        gym.Wrapper.__init__(self, env)
        if weight is None:
            weight = np.ones(shape=env.unwrapped.reward_space.shape)
        self.set_weight(weight)

    def set_weight(self, weight: np.ndarray):
        """Changes weights for the scalarization.

        Args:
            weight: new weights to set
        Returns: nothing
        """
        assert weight.shape == self.env.unwrapped.reward_space.shape, "Reward weight has different shape than reward vector."
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
        info["reward_weights"] = self.w

        return observation, scalar_reward, terminated, truncated, info


class MONormalizeReward(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Wrapper to normalize the reward component at index idx. Does not touch other reward components.

    This code is heavily inspired on Gymnasium's except that it extracts the reward component at given idx, normalizes it, and reinjects it.

    (!) This smoothes the moving average of the reward, which can be useful for training stability. But it does not "normalize" the reward in the sense of making it have a mean of 0 and a standard deviation of 1.

    Example:
        >>> import mo_gymnasium as mo_gym
        >>> from mo_gymnasium.wrappers import MONormalizeReward
        >>> env = mo_gym.make("deep-sea-treasure-v0")
        >>> norm_treasure_env = MONormalizeReward(env, idx=0)
        >>> both_norm_env = MONormalizeReward(norm_treasure_env, idx=1)
        >>> both_norm_env.reset() # This one normalizes both rewards

    """

    def __init__(self, env: gym.Env, idx: int, gamma: float = 0.99, epsilon: float = 1e-8):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            idx (int): the index of the reward to normalize
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.utils.RecordConstructorArgs.__init__(self, idx=idx, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        self.idx = idx
        self.return_rms = RunningMeanStd(shape=())
        self.discounted_reward: np.array = np.array([0.0])
        self.gamma = gamma
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the reward statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the reward statistics."""
        self._update_running_mean = setting

    def step(self, action: ActType):
        """Steps through the environment, normalizing the rewards returned.

        Args:
            action: action to perform
        Returns: obs, normalized_rewards, terminated, truncated, infos
        """
        obs, rews, terminated, truncated, infos = self.env.step(action)
        # Extracts the objective value to normalize
        to_normalize = rews[self.idx]

        self.discounted_reward = self.discounted_reward * self.gamma * (1 - terminated) + float(to_normalize)
        if self._update_running_mean:
            self.return_rms.update(self.discounted_reward)

        # We don't (reward - self.return_rms.mean) see https://github.com/openai/baselines/issues/538
        normalized_reward = to_normalize / np.sqrt(self.return_rms.var + self.epsilon)

        # Injecting the normalized objective value back into the reward vector
        rews[self.idx] = normalized_reward
        return obs, rews, terminated, truncated, infos


class MOClipReward(gym.RewardWrapper, gym.utils.RecordConstructorArgs):
    """Clip reward[idx] to [min, max]."""

    def __init__(self, env: gym.Env, idx: int, min_r, max_r):
        """Clip reward[idx] to [min, max].

        Args:
            env: environment to wrap
            idx: index of the MO reward to clip
            min_r: min reward
            max_r: max reward
        """
        gym.utils.RecordConstructorArgs.__init__(self, idx=idx, min_r=min_r, max_r=max_r)
        gym.RewardWrapper.__init__(self, env)
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


class MORecordEpisodeStatistics(RecordEpisodeStatistics, gym.utils.RecordConstructorArgs):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward (array)>",
        ...         "dr": "<discounted reward (array)>",
        ...         "l": "<episode length (scalar)>",
        ...         "t": "<elapsed time since beginning of episode (scalar)>"
        ...     },
        ... }
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 1.0,
        buffer_length: int = 100,
        stats_key: str = "episode",
    ):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            gamma (float): Discounting factor
            buffer_length: The size of the buffers :attr:`return_queue`, :attr:`length_queue` and :attr:`time_queue`
            stats_key: The info key for the episode statistics
        """
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, buffer_length=buffer_length, stats_key=stats_key)
        RecordEpisodeStatistics.__init__(self, env, buffer_length=buffer_length, stats_key=stats_key)
        # CHANGE: Here we just override the standard implementation to extend to MO
        self.reward_dim = self.env.unwrapped.reward_space.shape[0]
        self.rewards_shape = (self.reward_dim,)
        self.gamma = gamma

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        # This is very close the code from the RecordEpisodeStatistics wrapper from Gymnasium.
        (
            observation,
            rewards,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)
        assert isinstance(
            info, dict
        ), f"`info` dtype is {type(info)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards

        # CHANGE: The discounted returns are also computed here
        self.disc_episode_returns += rewards * np.repeat(self.gamma**self.episode_lengths, self.reward_dim).reshape(
            self.episode_returns.shape
        )
        self.episode_lengths += 1

        if terminated or truncated:
            assert self._stats_key not in info

            episode_time_length = round(time.perf_counter() - self.episode_start_time, 6)

            # Make a deepcopy to void subsequent mutation of the numpy array
            episode_returns = deepcopy(self.episode_returns)
            disc_episode_returns = deepcopy(self.disc_episode_returns)

            info["episode"] = {
                "r": episode_returns,
                "dr": disc_episode_returns,
                "l": self.episode_lengths,
                "t": episode_time_length,
            }

            self.time_queue.append(episode_time_length)
            self.return_queue.append(episode_returns)
            self.length_queue.append(self.episode_lengths)

            self.episode_count += 1
            self.episode_start_time = time.perf_counter()

        return (
            observation,
            rewards,
            terminated,
            truncated,
            info,
        )

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)

        # CHANGE: Here we just override the standard implementation to extend to MO
        self.episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)
        self.disc_episode_returns = np.zeros(self.rewards_shape, dtype=np.float32)

        return obs, info


class MOMaxAndSkipObservation(gym.Wrapper):
    """This wrapper will return only every ``skip``-th frame (frameskipping) and return the max between the two last observations.

    Note: This wrapper is based on the wrapper from stable-baselines3: https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#MaxAndSkipEnv
    """

    def __init__(self, env: gym.Env[ObsType, ActType], skip: int = 4):
        """This wrapper will return only every ``skip``-th frame (frameskipping) and return the max between the two last frames.

        Args:
            env (Env): The environment to apply the wrapper
            skip: The number of frames to skip
        """
        gym.Wrapper.__init__(self, env)

        if not np.issubdtype(type(skip), np.integer):
            raise TypeError(f"The skip is expected to be an integer, actual type: {type(skip)}")
        if skip < 2:
            raise ValueError(f"The skip value needs to be equal or greater than two, actual value: {skip}")
        if env.observation_space.shape is None:
            raise ValueError("The observation space must have the shape attribute.")

        self._skip = skip
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)

    def step(self, action):
        """Step the environment with the given action for ``skip`` steps.

        Repeat action, sum reward, and max over last observations.

        Args:
            action: The action to step through the environment with
        Returns:
            Max of the last two observations, reward, terminated, truncated, and info from the environment
        """
        total_reward = np.zeros(self.env.unwrapped.reward_dim, dtype=np.float32)
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

class RecordMarioVideo(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper records rollouts as videos.
    Allows intermittent recording of videos based on number of weights evaluted by specifying ``weight_trigger``.
    To increased weight_number, call `env.reset(options={"weights": w, "step":s})` at the beginning of each evaluation. 
    If weight trigger is activated, the video recorded file name will include the  current step `s` and evaluated weight `w` as a suffix. 
    `w` must be a numpy array and `s` must be an integer.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        disable_logger: bool = False,
        fps: int = 30,
    ):
        """Wrapper records rollouts as videos.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the videos will be stored
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            disable_logger (bool): Whether to disable logger or not.
            fps (int): Frames per second for the video recording.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        gym.Wrapper.__init__(self, env)

        if env.render_mode in {None, "human", "ansi", "ansi_list"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with"
                f" RecordVideo. Initialize your environment with a render_mode"
                f" that returns an image, such as rgb_array."
            )

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length
        self.fps = env.metadata.get("render_fps", fps)

        self.recording = False
        self.terminated = False
        self.truncated = False
        self.video_writer = None
        self.recorded_frames = 0
        self.episode_id = 0

        try:
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.is_vector_env = False

    def reset(self, **kwargs):
        """Reset the environment and start video recording if enabled."""
        observations = super().reset(**kwargs)
        self.terminated = False
        self.truncated = False
        if self.recording:
            assert self.video_writer is not None
            self._capture_frame()
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_writer()
        if self._video_enabled():
            self.start_video_recording()

        return observations

    def start_video_recording(self):
        """Initialize video recording."""
        self.close_video_writer()

        video_name = f"{self.name_prefix}-step-{self.step_id}.mp4"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}.mp4"
        video_path = os.path.join(self.video_folder, video_name)

        self.video_writer = imageio.get_writer(video_path, fps=self.fps, format='mp4')

        self._capture_frame()
        self.recording = True
    
    def _capture_frame(self):
        """Capture a frame from the environment and add it to the video."""
        frame = self.env.render()
        self.video_writer.append_data(frame)
        self.recorded_frames += 1

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        elif self.episode_trigger:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)

        if not (self.terminated or self.truncated):
            self.step_id += 1
            if not self.is_vector_env:
                if terminateds or truncateds:
                    self.episode_id += 1
                    self.terminated = terminateds
                    self.truncated = truncateds
            elif terminateds[0] or truncateds[0]:
                self.episode_id += 1
                self.terminated = terminateds[0]
                self.truncated = truncateds[0]

            if self.recording:
                assert self.video_writer is not None
                self._capture_frame()
                if self.video_length > 0:
                    if self.recorded_frames >= self.video_length:
                        self.close_video_writer()
                else:
                    if not self.is_vector_env:
                        if terminateds or truncateds:
                            self.close_video_writer()
                    elif terminateds[0] or truncateds[0]:
                        self.close_video_writer()
            elif self._video_enabled():
                self.start_video_recording()

        return observations, rewards, terminateds, truncateds, infos

    def close_video_writer(self):
        """Close the video writer if it is open."""
        if self.recording:
            assert self.video_writer is not None
            self.video_writer.close()
            self.recording = False
        self.recording = False
        self.recorded_frames = 1

    def close(self):
        """Closes the wrapper and saves any ongoing video recording."""
        super().close()
        self.close_video_writer()