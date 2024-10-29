import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle
from highway_env.envs import RacetrackEnv


class MORacetrackEnv(RacetrackEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the RacetrackEnv environment.

    See [highway-env](https://github.com/eleurent/highway-env) for more information.

    ## Reward Space
    The reward is 4-dimensional:
    - 0: lane centering reward
    - 1: action reward
    - 2: collision reward
    - 3: on road reward
    """

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)

        super().__init__(*args, **kwargs)
        self.reward_space = Box(
            low=np.array([0.0, 0.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 0.0, 1.0]), shape=(4,), dtype=np.float32
        )
        self.reward_dim = 4

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = info["rewards"]
        vec_reward = np.array(
            [
                rewards["lane_centering_reward"],
                rewards["action_reward"],
                -rewards["collision_reward"],
                rewards["on_road_reward"],
            ],
            dtype=np.float32,
        )
        info["original_reward"] = reward
        return obs, vec_reward, terminated, truncated, info
