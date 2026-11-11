from os import path
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle


BACKGROUND_COLOR = (250, 250, 246)
GRID_COLOR = (178, 204, 230)

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

CONVEX_FRONT = [
    np.array([0.7, -1]),
    np.array([8.2, -3]),
    np.array([11.5, -5]),
    np.array([14.0, -7]),
    np.array([15.1, -8]),
    np.array([16.1, -9]),
    np.array([19.6, -13]),
    np.array([20.3, -14]),
    np.array([22.4, -17]),
    np.array([23.7, -19]),
]

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

CONCAVE_FRONT = [
    np.array([1.0, -1]),
    np.array([2.0, -3]),
    np.array([3.0, -5]),
    np.array([5.0, -7]),
    np.array([8.0, -8]),
    np.array([16.0, -9]),
    np.array([24.0, -13]),
    np.array([50.0, -14]),
    np.array([74.0, -17]),
    np.array([124.0, -19]),
]

# As in Felten et al. 2022, same PF as concave, just harder map
MIRRORED_MAP = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -10, -10, 2.0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -10, -10, -10, -10, 3.0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -10, -10, -10, -10, -10, -10, 5.0, 8.0, 16.0, 0, 0, 0, 0],
        [0, 0, 0, 0, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0, 0],
        [0, 0, 0, 0, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0, 0],
        [0, 0, 0, 0, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 24.0, 50.0, 0, 0],
        [0, 0, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 0, 0],
        [0, 0, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 74.0, 0],
        [0, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 124.0],
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
    - treasure value: the value of the treasure at the current position
    - time penalty: -1 at each time step

    ## Starting State
    The starting state is always the same: (0, 0)

    ## Episode Termination
    The episode terminates when the agent reaches a treasure.

    ## Arguments
    - dst_map: the map of the deep sea treasure. Default is the convex map from Yang et al. (2019). To change, use `mo_gymnasium.make("DeepSeaTreasure-v0", dst_map=CONCAVE_MAP | MIRRORED_MAP).`
    - float_state: if True, the state is a 2D continuous box with values in [0.0, 1.0] for the x and y coordinates of the submarine.

    ## Credits
    The code was adapted from: [Yang's source](https://github.com/RunzheYang/MORL).
    The pixel art is provided by the Farama Foundation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, dst_map=DEFAULT_MAP, float_state=False):
        EzPickle.__init__(self, render_mode, dst_map, float_state)

        self.render_mode = render_mode
        self.float_state = float_state

        # The map of the deep sea treasure (convex version)
        self.sea_map = dst_map
        if dst_map.shape[0] == DEFAULT_MAP.shape[0] and dst_map.shape[1] == DEFAULT_MAP.shape[1]:
            if np.all(dst_map == DEFAULT_MAP):
                self.map_name = "convex"
            elif np.all(dst_map == CONCAVE_MAP):
                self.map_name = "concave"
            else:
                raise ValueError("Invalid map")
        elif np.all(dst_map == MIRRORED_MAP):
            self.map_name = "mirrored"
        else:
            raise ValueError("Invalid map")
        self._pareto_front = CONVEX_FRONT if self.map_name == "convex" else CONCAVE_FRONT

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
            self.observation_space = Box(low=0, high=len(self.sea_map[0]), shape=(2,), dtype=obs_type)

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_space = Discrete(4)
        self.reward_space = Box(
            low=np.array([0, -1]),
            high=np.array([np.max(self.sea_map), -1]),
            dtype=np.float32,
        )
        self.reward_dim = 2

        self.current_state = np.array([0, 0], dtype=np.int32)

        # pygame
        # The pixel art is drawn at 4x scale, so each grid square is 64x64 pixels
        self.cell_size = 64
        self.border = 32
        self.window_size = (
            self.sea_map.shape[1] * self.cell_size + 2 * self.border,
            self.sea_map.shape[0] * self.cell_size + 2 * self.border,
        )
        self.window = None
        self.clock = None
        self.sprites = None
        self.value_imgs = {}

    def pareto_front(self, gamma: float) -> List[np.ndarray]:
        """Return the discounted pareto front of the environment.

        Args:
            gamma: the discount factor.

        Returns:
            The discounted pareto front.

        """

        def discount_time(n):
            """Discounted time for a given number of steps."""
            return np.sum(np.array([gamma**i for i in range(int(n))]))

        # The first element is discounted based on the number of steps to reach there (which is -p[1])
        # e.g. if it takes 1 step to reach 0.7, the discounted value is 0.7 * gamma ** 0
        discounted_front = [np.array([p[0] * gamma ** (-p[1] - 1), -discount_time(-p[1])]) for p in self._pareto_front]
        return discounted_front

    def _get_map_value(self, pos):
        return self.sea_map[pos[0]][pos[1]]

    def _is_valid_state(self, state):
        if self.map_name == "mirrored":
            if state[0] >= 0 and state[0] <= 10 and state[1] >= 0 and state[1] <= 19:
                if self._get_map_value(state) != -10:
                    return True
            return False
        else:
            if state[0] >= 0 and state[0] <= 10 and state[1] >= 0 and state[1] <= 10:
                if self._get_map_value(state) != -10:
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
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Deep Sea Treasure")
                self.window = pygame.display.set_mode(self.window_size)
            else:
                self.window = pygame.Surface(self.window_size)

            if self.clock is None:
                self.clock = pygame.time.Clock()

        if self.sprites is None:
            self.sprites = {
                name: pygame.image.load(path.join(path.dirname(__file__), "assets", f"{name}.png"))
                for name in ["submarine", "tile_sea", "tile_sea_fleck", "tile_seabed", "treasure_chest"]
            }
            # The treasure value labels are pre-rendered, e.g. value_23_7.png (convex) or value_124.png (concave)
            for value in self.sea_map[self.sea_map > 0]:
                label = f"{value:.1f}".replace(".", "_") if self.map_name == "convex" else f"{int(value)}"
                filename = path.join(path.dirname(__file__), "assets", f"value_{label}.png")
                self.value_imgs[value] = pygame.image.load(filename)

        self.window.fill(BACKGROUND_COLOR)

        for i in range(self.sea_map.shape[0]):
            for j in range(self.sea_map.shape[1]):
                pos = np.array([j, i]) * self.cell_size + self.border
                value = self.sea_map[i, j]
                if value == 0:
                    tile = "tile_sea_fleck" if (j - i) % 4 == 0 else "tile_sea"
                    self.window.blit(self.sprites[tile], pos)
                else:
                    self.window.blit(self.sprites["tile_seabed"], pos)
                if value > 0:
                    self.window.blit(self.sprites["treasure_chest"], pos + np.array([12, 4]))
                    value_img = self.value_imgs[value]
                    # Labels are centered in the cell, excluding the 4-pixel grid line of the next cell
                    self.window.blit(value_img, pos + np.array([(self.cell_size - 4 - value_img.get_width()) // 2, 40]))

        # Tiles only draw their top and left grid lines, so close off the grid on the right and bottom
        grid_width = self.sea_map.shape[1] * self.cell_size
        grid_height = self.sea_map.shape[0] * self.cell_size
        pygame.draw.rect(self.window, GRID_COLOR, (self.border + grid_width - 4, self.border, 4, grid_height))
        pygame.draw.rect(self.window, GRID_COLOR, (self.border, self.border + grid_height - 4, grid_width, 4))

        submarine_img = self.sprites["submarine"]
        submarine_pos = self.current_state[::-1] * self.cell_size + self.border
        self.window.blit(submarine_img, submarine_pos + np.array([0, (self.cell_size - submarine_img.get_height()) // 2]))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def _get_state(self):
        if self.float_state:
            state = self.current_state.astype(np.float32) * 0.1
        else:
            state = self.current_state.copy()
        return state

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        if self.map_name == "convex" or self.map_name == "concave":
            self.current_state = np.array([0, 0], dtype=np.int32)
        elif self.map_name == "mirrored":
            self.current_state = np.array([0, 10], dtype=np.int32)
        else:
            raise ValueError("Invalid map")
        self.step_count = 0.0
        state = self._get_state()
        if self.render_mode == "human":
            self.render()
        return state, {}

    def step(self, action):
        next_state = self.current_state + self.dir[int(action)]

        if self._is_valid_state(next_state):
            self.current_state = next_state

        treasure_value = self._get_map_value(self.current_state)
        if treasure_value == 0 or treasure_value == -10:
            treasure_value = 0.0
            terminal = False
        else:
            terminal = True
        time_penalty = -1.0
        vec_reward = np.array([treasure_value, time_penalty], dtype=np.float32)

        state = self._get_state()
        if self.render_mode == "human":
            self.render()
        return state, vec_reward, terminal, False, {}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


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
