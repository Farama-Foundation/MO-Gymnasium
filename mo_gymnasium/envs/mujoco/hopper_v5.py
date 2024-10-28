import numpy as np
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOHopperEnv(HopperEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the HopperEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/hopper/) for more information.

    The original Gymnasium's 'Hopper-v5' is recovered by the following linear scalarization:

    env = mo_gym.make('mo-hopper-v5')
    LinearReward(env, weight=np.array([1.0, 0.0, 1e-3]))

    ## Reward Space
    The reward is 3-dimensional:
    - 0: Reward for going forward on the x-axis
    - 1: Reward for jumping high on the z-axis
    - 2: Control cost of the action
    If the cost_objective flag is set to False, the reward is 2-dimensional, and the cost is added to other objectives.

    A 2-objective version (without the cost objective as a separate objective) can be instantiated via:
    env = mo_gym.make('mo-hopper-2obj-v5')

    ## Version History
    - v5: The 2-objective version has now id 'mo-hopper-2obj-v5', instead of 'mo-hopper-2d-v4'.
    See https://gymnasium.farama.org/environments/mujoco/hopper/#version-history
    """

    def __init__(self, cost_objective=True, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, cost_objective, **kwargs)
        self._cost_objetive = cost_objective
        self.reward_dim = 3 if cost_objective else 2
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,))

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        x_velocity = info["x_velocity"]
        height = 10 * info["z_distance_from_origin"]
        neg_energy_cost = info["reward_ctrl"]
        if self._cost_objetive:
            neg_energy_cost /= self._ctrl_cost_weight  # Revert the scale applied in the original environment
            vec_reward = np.array([x_velocity, height, neg_energy_cost], dtype=np.float32)
        else:
            vec_reward = np.array([x_velocity, height], dtype=np.float32)
            vec_reward += neg_energy_cost

        vec_reward += info["reward_survive"]

        return observation, vec_reward, terminated, truncated, info
