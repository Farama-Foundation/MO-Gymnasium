from pathlib import Path
from typing import Optional

import gym
import numpy as np
import pygame
from gym.spaces import Box, Discrete


# As in Yang et al. (2019):
DEFAULT_MAP = np.array(
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

# As in Vamplew et al. (2018):
CONCAVE_MAP = np.array(
            [[0,    0,    0,   0,   0,  0,   0,   0,   0,   0,   0],
             [1.0,  0,    0,   0,   0,  0,   0,   0,   0,   0,   0],
             [-10,  2.0,  0,   0,   0,  0,   0,   0,   0,   0,   0],
             [-10, -10,  3.0,  0,   0,  0,   0,   0,   0,   0,   0],
             [-10, -10, -10, 5.0,  8.0,16.0, 0 ,  0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10,  0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10,  0,   0,   0,   0,   0],
             [-10, -10, -10, -10, -10, -10, 24.0, 50.0,0,   0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10,  0,   0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 74.0, 0,   0],
             [-10, -10, -10, -10, -10, -10, -10, -10, -10, 124.0,0]]
        )

class DeepSeaTreasure(gym.Env):
    """Deep Sea Treasure environment

    Adapted from: https://github.com/RunzheYang/MORL

    CCS weights: [1,0], [0.7,0.3], [0.67,0.33], [0.6,0.4], [0.56,0.44], [0.52,0.48], [0.5,0.5], [0.4,0.6], [0.3,0.7], [0, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, dst_map=DEFAULT_MAP, float_state=False):
        self.render_mode = render_mode
        self.size = 11
        self.window_size = 512
        self.window = None
        self.clock = None

        self.float_state = float_state

        # The map of the deep sea treasure (convex version)
        self.sea_map = dst_map
        assert self.sea_map.shape == DEFAULT_MAP.shape, "The map's shape must be 11x11"
        
        self.dir = {
            0: np.array([-1, 0], dtype=np.int32),  # up
            1: np.array([1, 0], dtype=np.int32),  # down
            2: np.array([0, -1], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32)  # right
        }

        # state space specification: 2-dimensional discrete box
        obs_type = np.float32 if self.float_state else np.int32
        if self.float_state:
            self.observation_space = Box(low=0.0, high=1.0, shape=(2,), dtype=obs_type)
        else:
            self.observation_space = Box(low=0, high=10, shape=(2,), dtype=obs_type)

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_space = Discrete(4)
        self.reward_space = Box(low=np.array([0, -1]), high=np.array([np.max(self.sea_map), -1]), dtype=np.float32)

        self.current_state = np.array([0, 0], dtype=np.int32)

    def get_map_value(self, pos):
        return self.sea_map[pos[0]][pos[1]]

    def is_valid_state(self, state):
        if state[0] >= 0 and state[0] <= 10 and state[1] >= 0 and state[1] <= 10:
            if self.get_map_value(state) != -10:
                return True
        return False
    
    def render(self):
        # The size of a single grid square in pixels
        pix_square_size = self.window_size / self.size
        if self.window is None:
            self.submarine_img = pygame.image.load(str(Path(__file__).parent.absolute()) + '/assets/submarine.png')
            self.submarine_img = pygame.transform.scale(self.submarine_img, (pix_square_size, pix_square_size))
            self.submarine_img = pygame.transform.flip(self.submarine_img, flip_x=True, flip_y=False)
            self.treasure_img = pygame.image.load(str(Path(__file__).parent.absolute()) + '/assets/treasure.png')
            self.treasure_img = pygame.transform.scale(self.treasure_img, (pix_square_size, pix_square_size))

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(None, 30)
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 105, 148))

        for i in range(self.sea_map.shape[0]):
            for j in range(self.sea_map.shape[1]):
                if self.sea_map[i,j] == -10:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.array([j,i]) + 0.6,
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.sea_map[i,j] != 0:
                   canvas.blit(self.treasure_img, np.array([j,i]) * pix_square_size)
                   img = self.font.render(str(self.sea_map[i,j]), True, (255, 255, 255))
                   canvas.blit(img, np.array([j,i]) * pix_square_size + np.array([5, 20]))
 
        canvas.blit(self.submarine_img, self.current_state[::-1] * pix_square_size)

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

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


if __name__ == '__main__':

    env = DeepSeaTreasure()
    done = False
    env.reset()
    while True:
        env.render()
        obs, r, done, info = env.step(env.action_space.sample())
        if done:
            env.reset()
