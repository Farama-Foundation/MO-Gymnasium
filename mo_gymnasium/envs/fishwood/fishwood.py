from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle


class FishWood(gym.Env, EzPickle):
    """
    ## Description
    The FishWood environment is a simple MORL problem in which the agent controls a fisherman which can either fish or go collect wood.
    From [Multi-objective Reinforcement Learning for the Expected Utility of the Return](https://www.researchgate.net/publication/328718263_Multi-objective_Reinforcement_Learning_for_the_Expected_Utility_of_the_Return).

    ## Observation Space
    The observation space is a discrete space with two states:
    - 0: fishing
    - 1: in the woods

    ## Action Space
    The actions is a discrete space where:
    - 0: go fishing
    - 1: go collect wood

    ## Reward Space
    The reward is 2-dimensional:
    - 0: +1 if agent is in the woods, with woodproba probability, and 0 otherwise
    - 1: +1 if the agent is fishing, with fishproba probability, and 0 otherwise

    ## Starting State
    Agent starts in the woods

    ## Termination
    The episode ends after MAX_TS=200 steps

    ## Arguments
    - fishproba: probability of catching a fish when fishing
    - woodproba: probability of collecting wood when in the woods

    ## Credits
    Code provided by Denis Steckelmacher
    """

    metadata = {"render_modes": ["human"]}
    FISH = np.array([0], dtype=np.int32)
    WOOD = np.array([1], dtype=np.int32)
    MAX_TS = 200

    def __init__(self, render_mode: Optional[str] = None, fishproba=0.1, woodproba=0.9):
        EzPickle.__init__(self, render_mode, fishproba, woodproba)

        self.render_mode = render_mode
        self._fishproba = fishproba
        self._woodproba = woodproba

        self.action_space = spaces.Discrete(2)  # 2 actions, go fish and go wood
        # 2 states, fishing and in the woods
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32)
        # 2 objectives, amount of fish and amount of wood
        self.reward_space = spaces.Box(low=np.array([0, 0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.reward_dim = 2

        self._state = self.WOOD.copy()

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self._state = self.WOOD.copy()
        self._timestep = 0
        if self.render_mode == "human":
            self.render()

        return self._state, {}

    def render(self):
        if self.render_mode == "human":
            if self._state == self.WOOD:
                return f"t={self._timestep}, in wood."
            else:
                return f"t={self._timestep}, fishing"

    def step(self, action):
        # Obtain a resource from the current state
        rewards = np.zeros((2,), dtype=np.float32)

        if self._state == self.WOOD and self.np_random.random() < self._woodproba:
            rewards[self.WOOD] = 1.0
        elif self._state == self.FISH and self.np_random.random() < self._fishproba:
            rewards[self.FISH] = 1.0

        # Execute the action
        self._state = np.array([action], dtype=np.int32)
        self._timestep += 1

        if self.render_mode == "human":
            self.render()
        return self._state, rewards, self._timestep == self.MAX_TS, self._timestep == self.MAX_TS, {}


if __name__ == "__main__":
    env = FishWood()
    terminated = False
    env.reset()
    while True:
        env.render()
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated:
            env.reset()
