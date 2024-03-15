import numpy as np
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOAntEnv(AntEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the AntEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/ant/) for more information.

    The original Gymnasium's 'Ant-v4' is recovered by the following linear scalarization:

    env = mo_gym.make('mo-ant-v4', cost_objective=False)
    LinearReward(env, weight=np.array([1.0, 0.0]))

    ## Reward Space
    The reward is 2- or 3-dimensional:
    - 0: x-velocity
    - 1: y-velocity
    - 2: Control cost of the action
    If the cost_objective flag is set to False, the reward is 2-dimensional, and the cost is added to other objectives.
    A healthy reward is added to all objectives.
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
        y_velocity = info["y_velocity"]
        cost = info["reward_ctrl"]
        healthy_reward = info["reward_survive"]

        if self.cost_objetive:
            cost /= self._ctrl_cost_weight  # Ignore the weight in the original AntEnv
            vec_reward = np.array([x_velocity, y_velocity, cost], dtype=np.float32)
        else:
            vec_reward = np.array([x_velocity, y_velocity], dtype=np.float32)
            vec_reward += cost

        vec_reward += healthy_reward

        return observation, vec_reward, terminated, truncated, info
