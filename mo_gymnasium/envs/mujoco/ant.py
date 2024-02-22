import numpy as np
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOAntEnv(AntEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the AntEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/ant/) for more information.

    ## Reward Space
    The reward is 2-dimensional:
    - 0: x-velocity
    - 1: y-velocity
    Both objectives contain cost and healthy penalties.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.reward_dim = 2

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        x_velocity = info["x_velocity"]
        y_velocity = info["y_velocity"]
        cost = info["reward_ctrl"]

        if self._use_contact_forces:
            cost -= self.contact_cost

        vec_reward = np.array([x_velocity, y_velocity], dtype=np.float32)
        vec_reward += cost

        return observation, vec_reward, terminated, truncated, info
