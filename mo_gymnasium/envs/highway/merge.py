import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from highway_env.envs import MergeEnv


class MOMergeEnv(MergeEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the MergeEnv environment.

    See [highway-env](https://github.com/eleurent/highway-env) for more information.

    ## Reward Space
    The reward is 5-dimensional:
    - 0: high speed reward
    - 1: right lane reward
    - 2: collision reward
    - 3: lane change reward
    - 4: merging speed reward
    """

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)

        super().__init__(*args, **kwargs)
        self.reward_space = Box(
            low=np.array([-1.0, 0.0, -1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 0.0, 1.0, 1.0]), shape=(5,), dtype=np.float32
        )
        self.reward_dim = 5

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = info["rewards"]
        vec_reward = np.array(
            [
                np.clip(rewards["high_speed_reward"], -1.0, 1.0),
                rewards["right_lane_reward"],
                -rewards["collision_reward"],
                rewards["lane_change_reward"],
                np.clip(rewards["merging_speed_reward"], 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        info["original_reward"] = reward
        return obs, vec_reward, terminated, truncated, info
