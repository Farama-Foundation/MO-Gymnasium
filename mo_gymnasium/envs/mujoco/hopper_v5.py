import numpy as np
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOHopperEnv(HopperEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the HopperEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/hopper/) for more information.

    ## Reward Space
    The reward is 3-dimensional:
    - 0: Reward for going forward on the x-axis
    - 1: Reward for jumping high on the z-axis
    - 2: Control cost of the action
    If the cost_objective flag is set to False, the reward is 2-dimensional, and the cost is added to other objectives.
    """

    def __init__(self, cost_objective=True, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, cost_objective, **kwargs)
        self.cost_objetive = cost_objective
        self.reward_dim = 3 if cost_objective else 2
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,))

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        x_velocity = info["x_velocity"]
        height = 10 * info["z_distance_from_origin"]
        energy_cost = np.sum(np.square(action))
        if self.cost_objetive:
            vec_reward = np.array([x_velocity, height, -energy_cost], dtype=np.float32)
        else:
            vec_reward = np.array([x_velocity, height], dtype=np.float32)
            vec_reward -= self._ctrl_cost_weight * energy_cost

        vec_reward += info["reward_survive"]

        return observation, vec_reward, terminated, truncated, info
