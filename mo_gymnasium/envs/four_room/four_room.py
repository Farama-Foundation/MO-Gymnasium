import random
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle


MAZE = np.array(
    [
        ["1", " ", " ", " ", " ", "2", "X", " ", " ", " ", " ", " ", "G"],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "1", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        ["2", " ", " ", " ", " ", "3", "X", " ", " ", " ", " ", " ", " "],
        ["X", "X", "3", " ", "X", "X", "X", "X", "X", " ", "1", "X", "X"],
        [" ", " ", " ", " ", " ", " ", "X", "2", " ", " ", " ", " ", "3"],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "2", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", "X", " ", " ", " ", " ", " ", " "],
        ["_", " ", " ", " ", " ", " ", "X", "3", " ", " ", " ", " ", "1"],
    ]
)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 128, 0)
BLACK = (0, 0, 0)


class FourRoom(gym.Env, EzPickle):
    """
    ## Description
    A discretized version of the gridworld environment introduced in [1]. Here, an agent learns to
    collect shapes with positive reward, while avoid those with negative reward, and then travel to a fixed goal.
    The gridworld is split into four rooms separated by walls with passage-ways.

    References
    ----------
    [1] Barreto, André, et al. "Successor Features for Transfer in Reinforcement Learning." NIPS. 2017.

    ## Observation Space
    The observation contains the 2D position of the agent in the gridworld, plus a binary vector indicating which items were collected.

    ## Action Space
    The action space is discrete with 4 actions: left, up, right, down.

    ## Reward Space
    The reward is a 3-dimensional vector with the following components:
    - +1 if collected a blue square, else 0
    - +1 if collected a green triangle, else 0
    - +1 if collected a red circle, else 0

    ## Starting State
    The agent starts in the lower left of the map.

    ## Episode Termination
    The episode terminates when the agent reaches the goal state, G.

    ## Arguments
    - maze: Array containing the gridworld map. See MAZE for an example.

    ## Credits
    Code adapted from: [Mike Gimelfarb's source](https://github.com/mike-gimelfarb/deep-successor-features-for-transfer/blob/main/source/tasks/gridworld.py).
    """

    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, maze=MAZE):
        """
        Creates a new instance of the shapes environment.

        Parameters
        ----------
        maze : np.ndarray
            an array of string values representing the type of each cell in the environment:
                G indicates a goal state (terminal state)
                _ indicates an initial state (there can be multiple, and one is selected at random
                    at the start of each episode)
                X indicates a barrier
                0, 1, .... 9 indicates the type of shape to be placed in the corresponding cell
                entries containing other characters are treated as regular empty cells
        """
        EzPickle.__init__(self, render_mode, maze)

        self.render_mode = render_mode
        self.window_size = int(13 * 35)
        self.window = None
        self.clock = None

        self.height, self.width = maze.shape
        self.maze = maze
        shape_types = ["1", "2", "3"]
        self.all_shapes = dict(zip(shape_types, range(len(shape_types))))

        self.goal = None
        self.initial = []
        self.occupied = set()
        self.shape_ids = dict()
        for c in range(self.width):
            for r in range(self.height):
                if maze[r, c] == "G":
                    self.goal = (r, c)
                elif maze[r, c] == "_":
                    self.initial.append((r, c))
                elif maze[r, c] == "X":
                    self.occupied.add((r, c))
                elif maze[r, c] in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                    self.shape_ids[(r, c)] = len(self.shape_ids)

        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=np.zeros(2 + len(self.shape_ids)),
            high=len(self.maze) * np.ones(2 + len(self.shape_ids)),
            dtype=np.int32,
        )
        self.reward_space = Box(low=0, high=1, shape=(3,))
        self.reward_dim = 3

    def state_to_array(self, state):
        s = [element for tupl in state for element in tupl]
        return np.array(s, dtype=np.int32)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self.state = (
            random.choice(self.initial),
            tuple(0 for _ in range(len(self.shape_ids))),
        )
        if self.render_mode == "human":
            self.render()
        return self.state_to_array(self.state), {}

    def step(self, action):
        old_state = self.state
        (row, col), collected = self.state

        # perform the movement
        if action == FourRoom.LEFT:
            col -= 1
        elif action == FourRoom.UP:
            row -= 1
        elif action == FourRoom.RIGHT:
            col += 1
        elif action == FourRoom.DOWN:
            row += 1
        else:
            raise Exception(f"bad action {action}")

        terminated = False

        # out of bounds, cannot move
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            return (
                self.state_to_array(self.state),
                np.zeros(len(self.all_shapes), dtype=np.float32),
                terminated,
                False,
                {},
            )

        # into a blocked cell, cannot move
        s1 = (row, col)
        if s1 in self.occupied:
            return (
                self.state_to_array(self.state),
                np.zeros(len(self.all_shapes), dtype=np.float32),
                terminated,
                False,
                {},
            )

        # can now move
        self.state = (s1, collected)

        # into a goal cell
        if s1 == self.goal:
            phi = np.ones(len(self.all_shapes), dtype=np.float32)
            terminated = True
            return self.state_to_array(self.state), phi, terminated, False, {}

        # into a shape cell
        if s1 in self.shape_ids:
            shape_id = self.shape_ids[s1]
            if collected[shape_id] == 1:
                # already collected this flag
                return (
                    self.state_to_array(self.state),
                    np.zeros(len(self.all_shapes), dtype=np.float32),
                    terminated,
                    False,
                    {},
                )
            else:
                # collect the new flag
                collected = list(collected)
                collected[shape_id] = 1
                collected = tuple(collected)
                self.state = (s1, collected)
                phi = self.features(old_state, action, self.state)
                return self.state_to_array(self.state), phi, terminated, False, {}

        if self.render_mode == "human":
            self.render()

        # into an empty cell
        return (
            self.state_to_array(self.state),
            np.zeros(len(self.all_shapes), dtype=np.float32),
            terminated,
            False,
            {},
        )

    def features(self, state, action, next_state):
        s1, _ = next_state
        _, collected = state
        nc = len(self.all_shapes)
        phi = np.zeros(nc, dtype=np.float32)
        if s1 in self.shape_ids:
            if collected[self.shape_ids[s1]] != 1:
                y, x = s1
                shape_index = self.all_shapes[self.maze[y, x]]
                phi[shape_index] = 1.0
        elif s1 == self.goal:
            phi[nc] = np.ones(nc, dtype=np.float32)
        return phi

    def render(self):
        # The size of a single grid square in pixels
        pix_square_size = self.window_size // 13

        if self.window is None and self.render_mode is not None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 48)
        img = self.font.render("G", True, BLACK)
        canvas.blit(img, (np.array(self.goal)[::-1] + 0.15) * pix_square_size)
        img = self.font.render("S", True, BLACK)
        canvas.blit(img, (np.array(self.initial[0])[::-1] + 0.15) * pix_square_size)

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                (row, col), collected = self.state
                shape_id = self.shape_ids.get((i, j), 0)
                if collected[shape_id] == 1 and self.maze[i, j] != "X":
                    continue

                pos = np.array([j, i])
                if self.maze[i, j] == "1":
                    pygame.draw.rect(
                        canvas,
                        BLUE,
                        pygame.Rect(
                            pix_square_size * pos,
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.maze[i, j] == "X":
                    pygame.draw.rect(
                        canvas,
                        BLACK,
                        pygame.Rect(
                            pix_square_size * pos + 1,
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif self.maze[i, j] == "2":
                    pygame.draw.polygon(
                        canvas,
                        GREEN,
                        [
                            (pos + np.array([0.5, 0.0])) * pix_square_size,
                            (pos + np.array([0.0, 1.0])) * pix_square_size,
                            (pos + 1.0) * pix_square_size,
                        ],
                    )
                elif self.maze[i, j] == "3":
                    pygame.draw.circle(
                        canvas,
                        RED,
                        (pos + 0.5) * pix_square_size,
                        pix_square_size / 2,
                    )

        pygame.draw.circle(
            canvas,
            (125, 125, 125),
            (np.array(self.state[0])[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for x in range(13 + 1):
            if x == 0 or x == 13:
                width = 3
                dash_lenght = 0
            else:
                width = 1
                dash_lenght = 3
            draw_line_dashed(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=width,
                dash_length=dash_lenght,
            )
            draw_line_dashed(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=width,
                dash_length=dash_lenght,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def draw_line_dashed(surface, color, start_pos, end_pos, width=1, dash_length=3, exclude_corners=True):
    """Code from https://codereview.stackexchange.com/questions/70143/drawing-a-dashed-line-with-pygame."""
    # convert tuples to numpy arrays
    if dash_length < 1:
        pygame.draw.line(surface, color, start_pos, end_pos, width)
    else:
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)

        # get euclidean distance between start_pos and end_pos
        length = np.linalg.norm(end_pos - start_pos)

        # get amount of pieces that line will be split up in (half of it are amount of dashes)
        dash_amount = int(length / dash_length)

        # x-y-value-pairs of where dashes start (and on next, will end)
        dash_knots = np.array([np.linspace(start_pos[i], end_pos[i], dash_amount) for i in range(2)]).transpose()

        return [
            pygame.draw.line(surface, color, tuple(dash_knots[n]), tuple(dash_knots[n + 1]), width)
            for n in range(int(exclude_corners), dash_amount - int(exclude_corners), 2)
        ]


if __name__ == "__main__":
    import mo_gymnasium as mo_gym

    env = mo_gym.make("four-room-v0", render_mode="human")
    terminated = False
    env.reset()
    while True:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
