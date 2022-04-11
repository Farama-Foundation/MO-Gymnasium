import gym
import numpy as np
from gym.spaces import Box, Discrete


class DeepSeaTreasure(gym.Env):
    """Deep Sea Treasure environment

    Adapted from: https://github.com/RunzheYang/MORL

    CCS weights: [1,0], [0.7,0.3], [0.67,0.33], [0.6,0.4], [0.56,0.44], [0.52,0.48], [0.5,0.5], [0.4,0.6], [0.3,0.7], [0, 1]
    """
    def __init__(self, float_state=False):

        self.float_state = float_state

        # The map of the deep sea treasure (convex version)
        self.sea_map = np.array(
            [[0,    0,    0,   0,   0,  0,   0,   0,   0,   0,   0],
             [0.7,  0,    0,   0,   0,  0,   0,   0,   0,   0,   0],
             [-10,  8.2,  0,   0,   0,  0,   0,   0,   0,   0,   0],
             [-10, -10, 11.5,  0,   0,  0,   0,   0,   0,   0,   0],
             [-10, -10, -10, 14.0, 15.1,16.1,0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10,  0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10,  0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10, 19.6, 20.3,0,   0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10,  0,   0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0]]
        )
        self.dir = {
            0: np.array([-1, 0], dtype=np.int32),  # up
            1: np.array([1, 0], dtype=np.int32),  # down
            2: np.array([0, -1], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32)  # right
        }

        # state space specification: 2-dimensional discrete box
        obs_type = np.float32 if self.float_state else np.int32
        self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=obs_type)

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_space = Discrete(4)
        self.reward_space = Box(low=np.array([0, -1]), high=np.array([23.7, -1]), dtype=np.float32)

        self.current_state = np.array([0, 0], dtype=np.int32)

    def get_map_value(self, pos):
        return self.sea_map[pos[0]][pos[1]]

    def is_valid_state(self, state):
        if state[0] >= 0 and state[0] <= 10 and state[1] >= 0 and state[1] <= 10:
            if self.get_map_value(state) != -10:
                return True
        return False
    
    def render(self, mode=None):
        pass

    def get_state(self):
        if self.float_state:
            state = self.current_state.astype(np.float32) * 0.1
        else:
            state = self.current_state.copy()
        return state

    def reset(self, **kwargs):
        self.current_state = np.array([0, 0], dtype=np.int32)
        self.step_count = 0.0
        state = self.get_state()
        return state

    def step(self, action):
        next_state = self.current_state + self.dir[action]

        if self.is_valid_state(next_state):
            self.current_state = next_state

        treasure_value = self.get_map_value(self.current_state)
        if treasure_value == 0 or treasure_value == -10:
            treasure_value = 0.0
            terminal = False
        else:
            terminal = True
        time_penalty = -1.0
        vec_reward = np.array([treasure_value, time_penalty], dtype=np.float32)

        state = self.get_state()

        return state, vec_reward, terminal, {}
