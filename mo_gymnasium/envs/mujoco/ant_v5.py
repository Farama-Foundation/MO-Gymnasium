import numpy as np
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOAntEnv(AntEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the AntEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/ant/) for more information.

    The original Gymnasium's 'Ant-v5' is recovered by the following linear scalarization:

    env = mo_gym.make('mo-ant-v4', cost_objective=False)
    LinearReward(env, weight=np.array([1.0, 0.0]))

    ## Reward Space
    The reward is 2- or 3-dimensional:
    - 0: x-velocity
    - 1: y-velocity
    - 2: Control cost of the action
    If the cost_objective flag is set to False, the reward is 2-dimensional, and the cost is added to other objectives.
    A healthy reward and a cost for contact forces is added to all objectives.

    A 2-objective version (without the cost objective as a separate objective) can be instantiated via:
    env = mo_gym.make('mo-ant-2obj-v5')

    ## Version History
    - v5: Now includes contact forces in the reward and observation.
          The 2-objective version has now id 'mo-ant-2obj-v5', instead of 'mo-ant-2d-v4'.
    See https://gymnasium.farama.org/environments/mujoco/ant/#version-history
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
        y_velocity = info["y_velocity"]
        cost = info["reward_ctrl"]
        healthy_reward = info["reward_survive"]

        if self._cost_objetive:
            cost /= self._ctrl_cost_weight  # Ignore the weight in the original AntEnv
            vec_reward = np.array([x_velocity, y_velocity, cost], dtype=np.float32)
        else:
            vec_reward = np.array([x_velocity, y_velocity], dtype=np.float32)
            vec_reward += cost

        vec_reward += healthy_reward
        vec_reward += info["reward_contact"]  # Do not treat contact forces as a separate objective

        return observation, vec_reward, terminated, truncated, info
