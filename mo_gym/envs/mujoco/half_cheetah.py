import numpy as np
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from gymnasium.spaces import Box


class MOHalfCheehtahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(2,))

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        vec_reward = np.array([info["reward_run"], info["reward_ctrl"]], dtype=np.float32)
        return observation, vec_reward, terminated, truncated, info
