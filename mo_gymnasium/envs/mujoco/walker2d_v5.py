import numpy as np
from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOWalker2dEnv(Walker2dEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the Walker2dEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/walker2d/) for more information.

    The original Gymnasium's 'Walker2d-v5' is recovered by the following linear scalarization:

    env = mo_gym.make('mo-walker2d-v5')
    LinearReward(env, weight=np.array([1.0, 1e-3]))

    ## Reward Space
    The reward is 2-dimensional:
    - 0: Reward for running forward (x-velocity)
    - 1: Control cost of the action

    ## Version History
    - See https://gymnasium.farama.org/main/environments/mujoco/walker2d/#version-history
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, **kwargs)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.reward_dim = 2

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        velocity = info["x_velocity"]
        neg_energy_cost = info["reward_ctrl"] / self._ctrl_cost_weight

        vec_reward = np.array([velocity, neg_energy_cost], dtype=np.float32)

        vec_reward += self.healthy_reward  # All objectives are penalyzed when the agent falls

        return observation, vec_reward, terminated, truncated, info
