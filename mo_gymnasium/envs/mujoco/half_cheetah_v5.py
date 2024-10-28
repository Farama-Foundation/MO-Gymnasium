import numpy as np
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOHalfCheehtahEnv(HalfCheetahEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the HalfCheetahEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) for more information.

    The original Gymnasium's 'HalfCheetah-v5' is recovered by the following linear scalarization:

    env = mo_gym.make('mo-halfcheetah-v4')
    LinearReward(env, weight=np.array([1.0, 0.1]))

    ## Reward Space
    The reward is 2-dimensional:
    - 0: Reward for running forward
    - 1: Control cost of the action

    ## Version History
    - v5: The scales of the control cost has changed from v4.
          See https://gymnasium.farama.org/environments/mujoco/half_cheetah/#version-history for other changes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.reward_dim = 2

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        x_velocity = info["x_velocity"]
        neg_energy_cost = info["reward_ctrl"] / self._ctrl_cost_weight  # Revert the scale applied in the original environment
        vec_reward = np.array([x_velocity, neg_energy_cost], dtype=np.float32)
        return observation, vec_reward, terminated, truncated, info
