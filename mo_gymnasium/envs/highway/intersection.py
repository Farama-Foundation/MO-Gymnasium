import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from highway_env.envs import IntersectionEnv


class MOIntersectionEnv(IntersectionEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the IntersectionEnv environment.

    See [highway-env](https://github.com/eleurent/highway-env) for more information.

    ## Reward Space
    The reward is 4-dimensional:
    - 0: high speed reward
    - 1: arrived reward
    - 2: collision reward
    - 3: on road reward
    """

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)

        super().__init__(*args, **kwargs)
        self.reward_space = Box(
            low=np.array([0.0, 0.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 0.0, 1.0]), shape=(4,), dtype=np.float64
        )
        self.reward_dim = 4

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = info["rewards"]
        vec_reward = np.array(
            [
                rewards["high_speed_reward"],
                rewards["arrived_reward"],
                -rewards["collision_reward"],
                rewards["on_road_reward"],
            ],
            dtype=np.float64,
        )
        info["original_reward"] = reward
        return obs, vec_reward, terminated, truncated, info
