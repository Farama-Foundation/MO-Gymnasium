import numpy as np
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOHumanoidEnv(HumanoidEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the HumanoidEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/humanoid/) for more information.

    ## Reward Space
    The reward is 2-dimensional:
    - 0: Reward for running forward (x-velocity)
    - 1: Control cost of the action
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.reward_dim = 2

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        velocity = info["x_velocity"]
        negative_cost = 10 * info["reward_quadctrl"]
        vec_reward = np.array([velocity, negative_cost], dtype=np.float32)

        vec_reward += self.healthy_reward  # All objectives are penalyzed when the agent falls

        return observation, vec_reward, terminated, truncated, info
