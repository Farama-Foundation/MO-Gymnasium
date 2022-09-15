from copy import deepcopy
from typing import Tuple, TypeVar, Iterator

import gym
import numpy as np
import time
from gym.vector import SyncVectorEnv
from gym.wrappers.normalize import RunningMeanStd
from gym.wrappers import RecordEpisodeStatistics

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def make(env_name: str, disable_env_checker: bool = True, **kwargs) -> gym.Env:
    """ Disable env checker, as it requires the reward to be a scalar."""
    return gym.make(env_name, disable_env_checker=disable_env_checker, **kwargs)


class LinearReward(gym.Wrapper):
    """Wrapper for Multi-Objective Envs
    Makes the env return a scalar reward, which is the the dot-product between the reward vector and the weight vector.
    """

    def __init__(self, env: gym.Env, weight: np.ndarray = None):
        super().__init__(env)
        if weight is None:
            weight = np.ones(shape=env.reward_space.shape)
        self.set_weight(weight)

    def set_weight(self, weight):
        assert weight.shape == self.env.reward_space.shape, "Reward weight has different shape than reward vector."
        self.w = weight

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        scalar_reward = np.dot(reward, self.w)
        info['vector_reward'] = reward

        return observation, scalar_reward, terminated, truncated, info


class MONormalizeReward(gym.Wrapper):
    """
    Wrapper to normalize the reward component at index idx. Does not touch other reward components.
    Based on Gym's implementation: https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py#L113
    """

    def __init__(self, env: gym.Env, idx: int, gamma: float = 0.99, epsilon: float = 1e-8):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

             Args:
                 env (env): The environment to apply the wrapper
                 idx (int): the index of the reward to normalize
                 epsilon (float): A stability parameter
                 gamma (float): The discount factor that is used in the exponential moving average.
             """
        super().__init__(env)
        self.idx = idx
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action: ActType):
        """Steps through the environment, normalizing the rewards returned."""
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
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)


class MOClipReward(gym.RewardWrapper):
    """"Clip reward[idx] to [min, max]. """

    def __init__(self, env: gym.Env, idx: int, min_r, max_r):
        super().__init__(env)
        self.idx = idx
        self.min_r = min_r
        self.max_r = max_r

    def reward(self, reward):
        reward[self.idx] = np.clip(reward[self.idx], self.min_r, self.max_r)
        return reward


class MOSyncVectorEnv(SyncVectorEnv):
    """Vectorized environment that serially runs multiple environments.
    """

    def __init__(
            self,
            env_fns: Iterator[callable],
            copy: bool = True,
    ):
        super().__init__(
            env_fns,
            copy=copy
        )
        # Just overrides the rewards memory to add the number of objectives
        self.reward_space = self.envs[0].reward_space
        self._rewards = np.zeros((self.num_envs, self.reward_space.shape[0],), dtype=np.float64)

def add_vector_episode_statistics(
    info: dict, episode_info: dict, num_envs: int, num_objs: int, env_num: int
):
    """Add episode statistics.

    Add statistics coming from the vectorized environment.

    Args:
        info (dict): info dict of the environment.
        episode_info (dict): episode statistics data.
        num_envs (int): number of environments.
        num_objs (int): number of objectives.
        env_num (int): env number of the vectorized environments.

    Returns:
        info (dict): the input info dict with the episode statistics.
    """
    info["episode"] = info.get("episode", {})

    info["_episode"] = info.get("_episode", np.zeros(num_envs, dtype=bool))
    info["_episode"][env_num] = True

    for k in episode_info.keys():
        if k == "r" or k == "dr":
            info_array = info["episode"].get(k, np.zeros((num_envs, num_objs)))
        else:
            info_array = info["episode"].get(k, np.zeros(num_envs))
        info_array[env_num] = deepcopy(episode_info[k])
        info["episode"][k] = info_array

    return info

class MORecordEpisodeStatistics(RecordEpisodeStatistics):
    def __init__(self, env: gym.Env, gamma: float = 1., deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        super().__init__(env, deque_size)
        # Here we just override the standard implementation to extend to MO
        self.reward_dim = self.env.reward_space.shape[0]
        self.gamma = gamma

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros((self.num_envs, self.reward_dim), dtype=np.float32)
        self.disc_episode_returns = np.zeros((self.num_envs, self.reward_dim), dtype=np.float32)
        return observations

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        # This is the code from the RecordEpisodeStatistics wrapper from gym.
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        # The discounted returns are also computed here
        self.disc_episode_returns += (rewards * np.repeat(self.gamma ** self.episode_lengths, self.reward_dim).reshape(self.episode_returns.shape))
        self.episode_lengths += 1
        if not self.is_vector_env:
            terminateds = [terminateds]
            truncateds = [truncateds]
        terminateds = list(terminateds)
        truncateds = list(truncateds)

        for i in range(len(terminateds)):
            if terminateds[i] or truncateds[i]:
                episode_return = deepcopy(self.episode_returns[i])  # Makes a deepcopy to avoid subsequent mutations
                disc_episode_return = deepcopy(self.disc_episode_returns[i])  # Makes a deepcopy to avoid subsequent mutations
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "episode": {
                        "r": episode_return,
                        "dr": disc_episode_return,
                        "l": episode_length,
                        "t": round(time.perf_counter() - self.t0, 6),
                    }
                }
                if self.is_vector_env:
                    infos = add_vector_episode_statistics(
                        infos, episode_info["episode"], self.num_envs, self.reward_dim, i
                    )
                else:
                    infos = {**infos, **episode_info}
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.disc_episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return (
            observations,
            rewards,
            terminateds if self.is_vector_env else terminateds[0],
            truncateds if self.is_vector_env else truncateds[0],
            infos,
        )

