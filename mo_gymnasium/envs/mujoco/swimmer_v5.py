import numpy as np
from gymnasium.envs.mujoco.swimmer_v5 import SwimmerEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOSwimmerEnv(SwimmerEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the SwimmerEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/swimmer/) for more information.

    The original Gymnasium's 'Swimmer-v4' is recovered by the following linear scalarization:

    env = mo_gym.make('mo-swimmer-v4')
    LinearReward(env, weight=np.array([1.0, 1e-4]))

    ## Reward Space
    The reward is 2-dimensional:
    - 0: Reward for moving forward (x-velocity)
    - 1: Control cost of the action

    ## Version History:
    See https://gymnasium.farama.org/main/environments/mujoco/swimmer/#version-history
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.reward_dim = 2

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        velocity = info["x_velocity"]
        neg_energy_cost = info["reward_ctrl"] / self._ctrl_cost_weight  # Revert the scale applied in the original environment

        vec_reward = np.array([velocity, neg_energy_cost], dtype=np.float32)

        return observation, vec_reward, terminated, truncated, info
