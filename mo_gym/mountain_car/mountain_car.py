import math
from typing import Optional

import numpy as np
from gym import spaces
from gym.envs.classic_control.mountain_car import MountainCarEnv


class MOMountainCar(MountainCarEnv):

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        super().__init__(render_mode, goal_velocity)

        self.reward_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
    
    def step(self, action: int):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        #reward = -1.0
        reward = np.zeros(3, dtype=np.float32)
        reward[0] = 0.0 if terminated else -1.0        # time penalty
        reward[1] = 0.0 if action != 0 else -1.0 # reverse penalty
        reward[2] = 0.0 if action != 2 else -1.0 # forward penalty

        self.state = (position, velocity)
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
