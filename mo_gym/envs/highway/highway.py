import numpy as np
from gym.spaces import Box
from highway_env.envs import HighwayEnv, HighwayEnvFast


class MOHighwayEnv(HighwayEnv):
    """A multi-objective version of the HighwayEnv environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = info["rewards"]
        vec_reward = np.array(
            [
                rewards["high_speed_reward"],
                rewards["right_lane_reward"],
                -rewards["collision_reward"],
            ],
            dtype=np.float32,
        )
        vec_reward *= rewards["on_road_reward"]
        info["original_reward"] = reward
        return obs, vec_reward, terminated, truncated, info


class MOHighwayEnvFast(HighwayEnvFast):
    """A multi-objective version of the HighwayFastEnv environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = info["rewards"]
        vec_reward = np.array(
            [
                rewards["high_speed_reward"],
                rewards["right_lane_reward"],
                -rewards["collision_reward"],
            ],
            dtype=np.float32,
        )
        vec_reward *= rewards["on_road_reward"]
        info["original_reward"] = reward
        return obs, vec_reward, terminated, truncated, info
