from typing import Tuple, TypeVar

import gym
import numpy as np
from gym.wrappers import NormalizeReward
from gym.wrappers.normalize import RunningMeanStd

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

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        observation, reward, done, info = self.env.step(action)
        scalar_reward = np.dot(reward, self.w)
        info['vector_reward'] = reward

        return observation, scalar_reward, done, info


class MONormalizeReward(gym.Wrapper):
    """
    Wrapper to normalize the reward component at index idx. Does not touch other reward components.
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
        obs, rews, dones, infos = self.env.step(action)
        to_normalize = rews[self.idx]
        if not self.is_vector_env:
            to_normalize = np.array([to_normalize])
        self.returns = self.returns * self.gamma + to_normalize
        to_normalize = self.normalize(to_normalize)
        self.returns[dones] = 0.0
        if not self.is_vector_env:
            to_normalize = to_normalize[0]
        rews[self.idx] = to_normalize
        return obs, rews, dones, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)


class MOClipReward(gym.RewardWrapper):
    r""""Clip reward to [min, max]. """

    def __init__(self, env: gym.Env, idx: int, min_r, max_r):
        super().__init__(env)
        self.idx = idx
        self.min_r = min_r
        self.max_r = max_r

    def reward(self, reward):
        reward[self.idx] = np.clip(reward[self.idx], self.min_r, self.max_r)
        return reward
