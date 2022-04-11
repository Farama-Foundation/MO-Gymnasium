from typing import Tuple, TypeVar

import gym
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LinearReward(gym.Wrapper):

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
