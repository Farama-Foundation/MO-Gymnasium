import math
from typing import Optional

import numpy as np
from gym import spaces
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv


class MOContinuousMountainCar(Continuous_MountainCarEnv):
    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        super().__init__(render_mode, goal_velocity)

        self.reward_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([0.0, 0.0]), shape=(2,), dtype=np.float32)

    def step(self, action: np.ndarray):
        # Essentially a copy paste from original env, except the rewards

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = np.zeros(2)
        # Time reward is negative at all timesteps except when reaching the goal
        if terminated:
            reward[0] = 0.0
        else:
            reward[0] = -1.0

        # Actions cost fuel, which we want to optimize too
        reward[1] = -math.pow(action[0], 2)

        self.state = np.array([position, velocity], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        return self.state, reward, terminated, False, {}
