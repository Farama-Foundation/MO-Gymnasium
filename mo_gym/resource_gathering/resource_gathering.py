from pathlib import Path

import gym
import numpy as np
import pygame
from gym.spaces import Box, Discrete


class ResourceGathering(gym.Env):
    """Resource Gathering environment

    From "Barrett, Leon & Narayanan, Srini. (2008). Learning all optimal policies with multiple criteria. 
    Proceedings of the 25th International Conference on Machine Learning. 41-47. 10.1145/1390156.1390162."
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        self.size = 5
        self.window_size = 512
        self.window = None
        self.clock = None

        # The map of resource gathering env
        self.map = np.array([
            [' ', ' ', 'R1', 'E2', ' '],
            [' ', ' ', 'E1', ' ', 'R2'],
            [' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' '],
            [' ', ' ', 'H', ' ', ' ']
            ]
        )
        self.initial_pos = np.array([4, 2], dtype=np.int32)

        self.dir = {
            0: np.array([-1, 0], dtype=np.int32),  # up
            1: np.array([1, 0], dtype=np.int32),  # down
            2: np.array([0, -1], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32)  # right
        }

        self.observation_space = Box(low=0.0, high=5.0, shape=(4,), dtype=np.int32)

        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_space = Discrete(4)
        # reward space: 
        self.reward_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def get_map_value(self, pos):
        return self.map[pos[0]][pos[1]]

    def is_valid_state(self, state):
        return state[0] >= 0 and state[0] < self.size and state[1] >= 0 and state[1] < self.size
    
    def render(self, mode='human'):
        # The size of a single grid square in pixels
        pix_square_size = self.window_size / self.size
        if self.window is None:
            self.gold_img = pygame.image.load(str(Path(__file__).parent.absolute()) + '/assets/gold.png')
            self.gold_img = pygame.transform.scale(self.gold_img, (pix_square_size, pix_square_size))
            self.gem_img = pygame.image.load(str(Path(__file__).parent.absolute()) + '/assets/gem.png')
            self.gem_img = pygame.transform.scale(self.gem_img, (pix_square_size, pix_square_size))
            self.enemy_img = pygame.image.load(str(Path(__file__).parent.absolute()) + '/assets/sword.png')
            self.enemy_img = pygame.transform.scale(self.enemy_img, (pix_square_size, pix_square_size))
            self.home_img = pygame.image.load(str(Path(__file__).parent.absolute()) + '/assets/home.png')
            self.home_img = pygame.transform.scale(self.home_img, (pix_square_size, pix_square_size))
            self.agent_img = pygame.image.load(str(Path(__file__).parent.absolute()) + '/assets/stickerman.png')
            self.agent_img = pygame.transform.scale(self.agent_img, (pix_square_size, pix_square_size))

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        canvas.blit(self.home_img, self.initial_pos[::-1] * pix_square_size)
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                pos = np.array([j,i])
                if self.map[i,j] == 'R1' and not self.has_gold:
                    canvas.blit(self.gold_img, np.array([j,i]) * pix_square_size)
                elif self.map[i,j] == 'R2' and not self.has_gem:
                    canvas.blit(self.gem_img, np.array([j,i]) * pix_square_size)
                elif self.map[i,j] == 'E1' or self.map[i,j] == 'E2':
                    canvas.blit(self.enemy_img, np.array([j,i]) * pix_square_size)
            
        canvas.blit(self.agent_img, self.current_pos[::-1] * pix_square_size)

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def get_state(self):
        pos = self.current_pos.copy()
        state = np.concatenate((pos, np.array([self.has_gold, self.has_gem], dtype=np.int32)))
        return state

    def reset(self, seed=None, return_info=False, **kwargs):
        super().reset(seed=seed)
        self.np_random.seed(seed)

        self.current_pos = self.initial_pos
        self.has_gem = 0
        self.has_gold = 0
        self.step_count = 0.0
        state = self.get_state()
        return (state, {}) if return_info else state

    def step(self, action):
        next_pos = self.current_pos + self.dir[action]

        if self.is_valid_state(next_pos):
            self.current_pos = next_pos

        vec_reward = np.zeros(3, dtype=np.float32)
        done = False

        cell = self.get_map_value(self.current_pos)
        if cell == 'R1':
            self.has_gold = 1
        elif cell == 'R2':
            self.has_gem = 1
        elif cell == 'E1' or cell == 'E2':
            if np.random.random() < 0.1:
                vec_reward[0] = -1.0
                done = True
        elif cell == 'H':
            done = True
            vec_reward[1] = self.has_gold
            vec_reward[2] = self.has_gem

        state = self.get_state()

        return state, vec_reward, done, {}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == '__main__':

    env = ResourceGathering()
    done = False
    env.reset()
    while True:
        env.render()
        obs, r, done, info = env.step(env.action_space.sample())
        print(obs, r, done)
        if done:
            env.reset()
