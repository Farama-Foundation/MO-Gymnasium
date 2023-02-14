from os import path
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle


# As in Yang et al. (2019):
DEFAULT_MAP = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, 8.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, -10, 11.5, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, -10, -10, 14.0, 15.1, 16.1, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 19.6, 20.3, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0],
    ]
)

# As in Vamplew et al. (2018):
CONCAVE_MAP = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, -10, 3.0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10, -10, -10, 5.0, 8.0, 16.0, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, 24.0, 50.0, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, 74.0, 0, 0],
        [-10, -10, -10, -10, -10, -10, -10, -10, -10, 124.0, 0],
    ]
)


class DeepSeaTreasure(gym.Env, EzPickle):
    """
    ## Description
    The Deep Sea Treasure environment is classic MORL problem in which the agent controls a submarine in a 2D grid world.

    ## Observation Space
    The observation space is a 2D discrete box with values in [0, 10] for the x and y coordinates of the submarine.

    ## Action Space
    The actions is a discrete space where:
    - 0: up
    - 1: down
    - 2: left
    - 3: right

    ## Reward Space
    The reward is 2-dimensional:
    - time penalty: -1 at each time step
    - treasure value: the value of the treasure at the current position

    ## Starting State
    The starting state is always the same: (0, 0)

    ## Episode Termination
    The episode terminates when the agent reaches a treasure.

    ## Arguments
    - dst_map: the map of the deep sea treasure. Default is the convex map from Yang et al. (2019). To change, use `mo_gymnasium.make("DeepSeaTreasure-v0", dst_map=CONCAVE_MAP).`
    - float_state: if True, the state is a 2D continuous box with values in [0.0, 1.0] for the x and y coordinates of the submarine.

    ## Credits
    The code was adapted from: [Yang's source](https://github.com/RunzheYang/MORL).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, dst_map=DEFAULT_MAP, float_state=False):
        EzPickle.__init__(self, render_mode, dst_map, float_state)

        self.render_mode = render_mode
        self.float_state = float_state

        # The map of the deep sea treasure (convex version)
        self.sea_map = dst_map
        assert self.sea_map.shape == DEFAULT_MAP.shape, "The map's shape must be 11x11"

        self.dir = {
            0: np.array([-1, 0], dtype=np.int32),  # up
            1: np.array([1, 0], dtype=np.int32),  # down
            2: np.array([0, -1], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32),  # right
        }

        # state space specification: 2-dimensional discrete box
        obs_type = np.float32 if self.float_state else np.int32
        if self.float_state:
            self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=obs_type)
        else:
            self.observation_space = Box(low=0, high=10, shape=(2,), dtype=obs_type)

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_space = Discrete(4)
        self.reward_space = Box(
            low=np.array([0, -1]),
            high=np.array([np.max(self.sea_map), -1]),
            dtype=np.float32,
        )

        self.current_state = np.array([0, 0], dtype=np.int32)

        # pygame
        self.window_size = (min(64 * self.sea_map.shape[1], 512), min(64 * self.sea_map.shape[0], 512))
        # The size of a single grid square in pixels
        self.pix_square_size = (
            self.window_size[1] // self.sea_map.shape[1] + 1,
            self.window_size[0] // self.sea_map.shape[0] + 1,
        )
        self.window = None
        self.clock = None
        self.submarine_img = None
        self.treasure_img = None
        self.sea_img = None
        self.rock_img = None

    def get_map_value(self, pos):
        return self.sea_map[pos[0]][pos[1]]

    def is_valid_state(self, state):
        if state[0] >= 0 and state[0] <= 10 and state[1] >= 0 and state[1] <= 10:
            if self.get_map_value(state) != -10:
                return True
        return False

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. mo_gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.window is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Deep Sea Treasure")
                self.window = pygame.display.set_mode(self.window_size)
            else:
                self.window = pygame.Surface(self.window_size)

            if self.clock is None:
                self.clock = pygame.time.Clock()

            if self.submarine_img is None:
                filename = path.join(path.dirname(__file__), "assets", "submarine.png")
                self.submarine_img = pygame.transform.scale(pygame.image.load(filename), self.pix_square_size)
                self.submarine_img = pygame.transform.flip(self.submarine_img, flip_x=True, flip_y=False)
            if self.treasure_img is None:
                filename = path.join(path.dirname(__file__), "assets", "treasure.png")
                self.treasure_img = pygame.transform.scale(pygame.image.load(filename), self.pix_square_size)
            if self.sea_img is None:
                filename = path.join(path.dirname(__file__), "assets", "sea.png")
                self.sea_img = pygame.transform.scale(pygame.image.load(filename), self.pix_square_size)
            if self.rock_img is None:
                filename = path.join(path.dirname(__file__), "assets", "rock.png")
                self.rock_img = pygame.transform.scale(pygame.image.load(filename), self.pix_square_size)

            self.font = pygame.font.Font(path.join(path.dirname(__file__), "assets", "Minecraft.ttf"), 20)

        for i in range(self.sea_map.shape[0]):
            for j in range(self.sea_map.shape[1]):
                if self.sea_map[i, j] == -10:
                    self.window.blit(self.rock_img, np.array([j, i]) * self.pix_square_size)
                elif self.sea_map[i, j] == 0:
                    self.window.blit(self.sea_img, np.array([j, i]) * self.pix_square_size)
                elif self.sea_map[i, j] != 0:
                    self.window.blit(self.sea_img, np.array([j, i]) * self.pix_square_size)
                    self.window.blit(self.treasure_img, np.array([j, i]) * self.pix_square_size)
                    trailing_space = " " if self.sea_map[i, j] < 10 else ""
                    img = self.font.render(trailing_space + str(self.sea_map[i, j]), True, (255, 255, 255))
                    self.window.blit(img, np.array([j, i]) * self.pix_square_size + np.array([5, -20]))

        self.window.blit(self.submarine_img, self.current_state[::-1] * self.pix_square_size)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def get_state(self):
        if self.float_state:
            state = self.current_state.astype(np.float32) * 0.1
        else:
            state = self.current_state.copy()
        return state

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self.current_state = np.array([0, 0], dtype=np.int32)
        self.step_count = 0.0
        state = self.get_state()
        if self.render_mode == "human":
            self.render()
        return state, {}

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
        if self.render_mode == "human":
            self.render()
        return state, vec_reward, terminal, False, {}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    import mo_gymnasium as mo_gym

    env = mo_gym.make("deep-sea-treasure-v0", render_mode="human")
    terminated = False
    env.reset()
    while True:
        env.render()
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
