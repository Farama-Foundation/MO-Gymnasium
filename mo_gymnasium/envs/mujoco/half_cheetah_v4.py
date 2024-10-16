import numpy as np
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOHalfCheehtahEnv(HalfCheetahEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the HalfCheetahEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) for more information.

    The original Gymnasium's 'HalfCheetah-v4' is recovered by the following linear scalarization:

    env = mo_gym.make('mo-halfcheetah-v4')
    LinearReward(env, weight=np.array([1.0, 1.0]))

    ## Reward Space
    The reward is 2-dimensional:
    - 0: Reward for running forward
    - 1: Control cost of the action
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.reward_dim = 2

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        vec_reward = np.array([info["reward_run"], info["reward_ctrl"]], dtype=np.float32)
        return observation, vec_reward, terminated, truncated, info
