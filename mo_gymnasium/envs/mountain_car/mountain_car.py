import math
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.utils import EzPickle


class MOMountainCar(MountainCarEnv, EzPickle):
    """
    A multi-objective version of the MountainCar environment, where the goal is to reach the top of the mountain.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) for more information.

    ## Reward space:
    By default, the reward space is a 3D vector containing the time penalty, and penalties for reversing and going forward.
    - time penalty: -1.0 for each time step
    - reverse penalty: -1.0 for each time step the action is 0 (reverse)
    - forward penalty: -1.0 for each time step the action is 2 (forward)

    #Alternatively, the reward can be changed with the following options:
    - add_speed_objective: Add an extra objective corresponding to the speed of the car.
    - remove_move_penalty: Remove the reverse and forward objectives.
    - merge_move_penalty: Merge reverse and forward penalties into a single penalty.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        add_speed_objective: bool = False,
        remove_move_penalty: bool = False,
        merge_move_penalty: bool = False,
        goal_velocity=0,
    ):
        super().__init__(render_mode, goal_velocity)
        EzPickle.__init__(self, render_mode, add_speed_objective, remove_move_penalty, merge_move_penalty, goal_velocity)
        self.add_speed_objective = add_speed_objective
        self.remove_move_penalty = remove_move_penalty
        self.merge_move_penalty = merge_move_penalty

        self.reward_dim = 3

        if self.add_speed_objective:
            self.reward_dim += 1

        if self.remove_move_penalty:
            self.reward_dim -= 2
        elif self.merge_move_penalty:
            self.reward_dim -= 1

        low = np.array([-1] * self.reward_dim)
        high = np.zeros(self.reward_dim)
        high[0] = -1  # Time penalty is always -1
        if self.add_speed_objective:
            low[-1] = 0.0
            high[-1] = 1.1

        self.reward_space = spaces.Box(low=low, high=high, shape=(self.reward_dim,), dtype=np.float32)

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

        reward = np.zeros(self.reward_dim, dtype=np.float32)

        reward[0] = 0.0 if terminated else -1.0  # time penalty

        if not self.remove_move_penalty:
            if self.merge_move_penalty:
                reward[1] = 0.0 if action == 1 else -1.0
            else:
                reward[1] = 0.0 if action != 0 else -1.0  # reverse penalty
                reward[2] = 0.0 if action != 2 else -1.0  # forward penalty

        if self.add_speed_objective:
            reward[-1] = 15 * abs(velocity)

        self.state = (position, velocity)
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
