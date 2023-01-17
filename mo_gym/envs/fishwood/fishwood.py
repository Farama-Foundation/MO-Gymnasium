import random
from typing import Optional

import gym
import numpy as np
from gym import spaces


class FishWood(gym.Env):
    metadata = {"render_modes": ["ansi"]}
    FISH = 0
    WOOD = 1
    MAX_TS = 200

    def __init__(self, render_mode: Optional[str] = None, fishproba=0.1, woodproba=0.9):
        self.render_mode = render_mode
        self._fishproba = fishproba
        self._woodproba = woodproba

        self.action_space = spaces.Discrete(2)  # 2 actions, go fish and go wood
        # 2 states, fishing and in the woods
        self.observation_space = spaces.Discrete(2)
        # 2 objectives, amount of fish and amount of wood
        self.reward_space = spaces.Box(low=np.array([0, 0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self._state = self.reset()

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self._state = self.WOOD
        self._timestep = 0
        if self.render_mode == "human":
            self.render()

        return self._state, {}

    def render(self):
        if self._state == self.WOOD:
            return f"t={self._timestep}, in wood."
        else:
            return f"t={self._timestep}, fishing"

    def step(self, action):
        # Obtain a resource from the current state
        rewards = np.zeros((2,))

        if self._state == self.WOOD and random.random() < self._woodproba:
            rewards[self.WOOD] = 1.0
        elif self._state == self.FISH and random.random() < self._fishproba:
            rewards[self.FISH] = 1.0

        # Execute the action
        self._state = action
        self._timestep += 1

        return self._state, rewards, self._timestep == self.MAX_TS, False, {}


if __name__ == "__main__":

    env = FishWood()
    terminated = False
    env.reset()
    while True:
        env.render()
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated:
            env.reset()
