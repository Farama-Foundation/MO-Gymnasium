import itertools
import json
import math
from math import ceil
from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pygame
import scipy.stats
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle
from scipy.spatial import ConvexHull


EPS_SPEED = 0.001  # Minimum speed to be considered in motion
HOME_X = 0.0
HOME_Y = 0.0
HOME_POS = (HOME_X, HOME_Y)

ROTATION = 10
MAX_SPEED = 1.0

FUEL_MINE = -0.05
FUEL_ACC = -0.025
FUEL_IDLE = -0.005

CAPACITY = 1

ACT_MINE = 0
ACT_LEFT = 1
ACT_RIGHT = 2
ACT_ACCEL = 3
ACT_BRAKE = 4
ACT_NONE = 5
FUEL_LIST = [
    FUEL_MINE + FUEL_IDLE,
    FUEL_IDLE,
    FUEL_IDLE,
    FUEL_IDLE + FUEL_ACC,
    FUEL_IDLE,
    FUEL_IDLE,
]
FUEL_DICT = {
    ACT_MINE: FUEL_MINE + FUEL_IDLE,
    ACT_LEFT: FUEL_IDLE,
    ACT_RIGHT: FUEL_IDLE,
    ACT_ACCEL: FUEL_IDLE + FUEL_ACC,
    ACT_BRAKE: FUEL_IDLE,
    ACT_NONE: FUEL_IDLE,
}
ACTIONS = ["Mine", "Left", "Right", "Accelerate", "Brake", "None"]
ACTION_COUNT = len(ACTIONS)


MINE_RADIUS = 0.14
BASE_RADIUS = 0.15

WIDTH = 480
HEIGHT = 480

# Color definitions
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
C_GRAY = (150 / 255.0, 150 / 255.0, 150 / 255.0)
DARK_GRAY = (100, 100, 100)
BLACK = (0, 0, 0)
RED = (255, 70, 70)
C_RED = (1.0, 70 / 255.0, 70 / 255.0)

FPS = 180

MINE_LOCATION_TRIES = 100

MINE_SCALE = 1.0
BASE_SCALE = 1.0
CART_SCALE = 1.0

MARGIN = 0.16 * CART_SCALE

ACCELERATION = 0.0075 * CART_SCALE
DECELERATION = 1

CART_IMG = str(Path(__file__).parent.absolute()) + "/assets/cart.png"
MINE_IMG = str(Path(__file__).parent.absolute()) + "/assets/mine.png"


class Minecart(gym.Env, EzPickle):
    """
    ## Description
    Agent must collect two types of ores and minimize fuel consumption.
    From [Abels et al. 2019](https://arxiv.org/abs/1809.07803v2).

    ## Observation Space
    The observation is a 7-dimensional vector containing the following information:
    - 2D position of the cart
    - Speed of the cart
    - sin and cos of the cart's orientation
    - porcentage of the capacity of the cart filled
    If image_observation is True, the observation is a 3D image of the environment.

    ## Action Space
    The action space is a discrete space with 6 actions:
    - 0: Mine
    - 1: Left
    - 2: Right
    - 3: Accelerate
    - 4: Brake
    - 5: None

    ## Reward Space
    The reward is a 3D vector:
    - 0: Quantity of the first minerium that was retrieved to the base (sparse)
    - 1: Quantity of the second minerium that was retrieved to the base (sparse)
    - 2: Fuel consumed (dense)

    ## Starting State
    The cart starts at the base on the upper left corner of the map.

    ## Episode Termination
    The episode ends when the cart returns to the base.

    ## Arguments
    - render_mode: The render mode to use. Can be "rgb_array" or "human".
    - image_observation: If True, the observation is a RGB image of the environment.
    - frame_skip: How many times each action is repeated. Default: 4
    - incremental_frame_skip: Whether actions are repeated incrementally. Default: True
    - config: Path to the .json configuration file. See the default configuration file for more information: https://github.com/Farama-Foundation/MO-Gymnasium/blob/main/mo_gymnasium/envs/minecart/mine_config.json

    ## Credits
    The code was refactored from [Axel Abels' source](https://github.com/axelabels/DynMORL).
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        image_observation: bool = False,
        frame_skip: int = 4,
        incremental_frane_skip: bool = True,
        config=str(Path(__file__).parent.absolute()) + "/mine_config.json",
    ):
        EzPickle.__init__(self, render_mode, image_observation, frame_skip, incremental_frane_skip, config)

        self.render_mode = render_mode
        self.screen = None
        self.canvas = None
        self.clock = None
        self.last_render_mode_used = None
        self.config = config
        self.frame_skip = frame_skip
        assert self.frame_skip > 0, "Frame skip must be greater than 0."
        self.incremental_frame_skip = incremental_frane_skip

        with open(self.config) as f:
            data = json.load(f)

        self.ore_cnt = data["ore_cnt"]
        self.capacity = data["capacity"]
        self.mine_cnt = data["mine_cnt"]
        ore_colors = None if "ore_colors" not in data else data["ore_colors"]
        self.ore_colors = ore_colors or [
            (
                np.random.randint(40, 255),
                np.random.randint(40, 255),
                np.random.randint(40, 255),
            )
            for i in range(self.ore_cnt)
        ]
        self.generate_mines(None)

        if "mines" in data:
            for mine_data, mine in zip(data["mines"], self.mines):
                mine.pos = np.array([mine_data["x"], mine_data["y"]])
                if "distributions" in mine_data:
                    mine.distributions = [scipy.stats.norm(dist[0], dist[1]) for dist in mine_data["distributions"]]

        self.cart = Cart(self.ore_cnt)

        self.end = False

        self.image_observation = image_observation
        if self.image_observation:
            shape = (WIDTH, HEIGHT, 3)
            self.observation_space = Box(
                low=np.zeros(shape),
                high=255 * np.ones((WIDTH, HEIGHT, 3)),
                dtype=np.uint8,
            )
        else:
            self.observation_space = Box(-np.ones(7), np.ones(7), dtype=np.float32)

        self.action_space = Discrete(6)
        self.reward_space = Box(
            low=np.append(np.zeros(self.ore_cnt), -1.0),
            high=np.append(np.ones(self.ore_cnt) * self.capacity, 0.0),
            shape=(self.ore_cnt + 1,),
        )
        self.reward_dim = self.ore_cnt + 1

    def convex_coverage_set(self, gamma: float, symmetric: bool = True) -> List[np.ndarray]:
        """
        Computes an approximate convex coverage set (CCS).

        Args:
            gamma (float): Discount factor to apply to rewards.
            symmetric (bool): If true, we assume the pattern of accelerations from the base to the mine is the same as from the mine to the base. Default: True

        Returns:
            The convex coverage set
        """
        policies = self.pareto_front(gamma, symmetric)
        origin = np.min(policies, axis=0)
        extended_policies = [origin] + policies
        return [policies[idx - 1] for idx in ConvexHull(extended_policies).vertices if idx != 0]

    def pareto_front(self, gamma: float, symmetric: bool = True) -> List[np.ndarray]:
        """
        Computes an approximate pareto front.

        Args:
            gamma (float): Discount factor to apply to rewards
            symmetric (bool): If true, we assume the pattern of accelerations from the base to the mine is the same as from the mine to the base. Default: True

        Returns:
            The pareto coverage set
        """
        all_rewards = []
        base_perimeter = BASE_RADIUS * BASE_SCALE

        # Empty mine just outside the base
        virtual_mine = Mine(
            self.ore_cnt,
            (base_perimeter**2 / 2) ** (1 / 2),
            (base_perimeter**2 / 2) ** (1 / 2),
        )
        virtual_mine.distributions = [scipy.stats.norm(0, 0) for _ in range(self.ore_cnt)]
        for mine in self.mines + [virtual_mine]:
            mine_distance = mag(mine.pos - HOME_POS) - MINE_RADIUS * MINE_SCALE - BASE_RADIUS * BASE_SCALE / 2

            # Number of rotations required to face the mine
            angle = compute_angle(mine.pos, HOME_POS, [1, 1])
            rotations = int(ceil(abs(angle) / (ROTATION * self.frame_skip)))

            # Build pattern of accelerations/nops to reach the mine
            # initialize with single acceleration
            queue = [
                {
                    "speed": ACCELERATION * self.frame_skip,
                    "dist": (
                        mine_distance - self.frame_skip * (self.frame_skip + 1) / 2 * ACCELERATION
                        if self.incremental_frame_skip
                        else mine_distance - ACCELERATION * self.frame_skip * self.frame_skip
                    ),
                    "seq": [ACT_ACCEL],
                }
            ]
            trimmed_sequences = []

            while len(queue) > 0:
                seq = queue.pop()
                # accelerate
                new_speed = seq["speed"] + ACCELERATION * self.frame_skip
                accelerations = new_speed / ACCELERATION
                movement = (
                    accelerations * (accelerations + 1) / 2 * ACCELERATION
                    - (accelerations - self.frame_skip) * ((accelerations - self.frame_skip) + 1) / 2 * ACCELERATION
                )
                dist = seq["dist"] - movement
                speed = new_speed
                if dist <= 0:
                    trimmed_sequences.append(seq["seq"] + [ACT_ACCEL])
                else:
                    queue.append({"speed": speed, "dist": dist, "seq": seq["seq"] + [ACT_ACCEL]})
                # idle
                dist = seq["dist"] - seq["speed"] * self.frame_skip

                if dist <= 0:
                    trimmed_sequences.append(seq["seq"] + [ACT_NONE])
                else:
                    queue.append(
                        {
                            "speed": seq["speed"],
                            "dist": dist,
                            "seq": seq["seq"] + [ACT_NONE],
                        }
                    )

            # Build rational mining sequences
            mine_means = mine.distribution_means() * self.frame_skip
            mn_sum = np.sum(mine_means)
            # on average it takes up to this many actions to fill cart
            max_mine_actions = 0 if mn_sum == 0 else int(ceil(self.capacity / mn_sum))

            # all possible mining sequences (i.e. how many times we mine)
            mine_sequences = [[ACT_MINE] * i for i in range(1, max_mine_actions + 1)]

            # All possible combinations of actions before, during and after mining
            if len(mine_sequences) > 0:
                if not symmetric:
                    all_sequences = map(
                        lambda sequences: list(sequences[0])
                        + list(sequences[1])
                        + list(sequences[2])
                        + list(sequences[3])
                        + list(sequences[4]),
                        itertools.product(
                            [[ACT_LEFT] * rotations],
                            trimmed_sequences,
                            [[ACT_BRAKE] + [ACT_LEFT] * (180 // (ROTATION * self.frame_skip))],
                            mine_sequences,
                            trimmed_sequences,
                        ),
                    )

                else:
                    all_sequences = map(
                        lambda sequences: list(sequences[0])
                        + list(sequences[1])
                        + list(sequences[2])
                        + list(sequences[3])
                        + list(sequences[1]),
                        itertools.product(
                            [[ACT_LEFT] * rotations],
                            trimmed_sequences,
                            [[ACT_BRAKE] + [ACT_LEFT] * (180 // (ROTATION * self.frame_skip))],
                            mine_sequences,
                        ),
                    )
            else:
                if not symmetric:
                    print(
                        [ACT_NONE] + trimmed_sequences[1:],
                        trimmed_sequences[1:],
                        trimmed_sequences,
                    )
                    all_sequences = map(
                        lambda sequences: list(sequences[0])
                        + list(sequences[1])
                        + list(sequences[2])
                        + [ACT_NONE]
                        + list(sequences[3])[1:],
                        itertools.product(
                            [[ACT_LEFT] * rotations],
                            trimmed_sequences,
                            [[ACT_LEFT] * (180 // (ROTATION * self.frame_skip))],
                            trimmed_sequences,
                        ),
                    )

                else:
                    all_sequences = map(
                        lambda sequences: list(sequences[0])
                        + list(sequences[1])
                        + list(sequences[2])
                        + [ACT_NONE]
                        + list(sequences[1][1:]),
                        itertools.product(
                            [[ACT_LEFT] * rotations],
                            trimmed_sequences,
                            [[ACT_LEFT] * (180 // (ROTATION * self.frame_skip))],
                        ),
                    )

            # Compute rewards for each sequence
            fuel_costs = np.array([f * self.frame_skip for f in FUEL_LIST])

            def maxlen(l):
                if len(l) == 0:
                    return 0
                return max([len(s) for s in l])

            longest_pattern = maxlen(trimmed_sequences)
            max_len = (
                rotations
                + longest_pattern
                + 1
                + (180 // (ROTATION * self.frame_skip))
                + maxlen(mine_sequences)
                + longest_pattern
            )
            discount_map = gamma ** np.arange(max_len)
            for s in all_sequences:
                reward = np.zeros((len(s), self.reward_dim))
                reward[:, -1] = fuel_costs[s]
                mine_actions = s.count(ACT_MINE)
                reward[-1, :-1] = mine_means * mine_actions / max(1, (mn_sum * mine_actions) / self.capacity)

                reward = np.dot(discount_map[: len(s)], reward)
                all_rewards.append(reward)

            all_rewards = pareto_filter(all_rewards, minimize=False)

        return all_rewards

    def generate_mines(self, mine_distributions=None):
        """
        Randomly generate mines that don't overlap the base
        TODO: propose some default formations
        """
        self.mines = []
        for i in range(self.mine_cnt):
            pos = np.array((np.random.random(), np.random.random()))

            tries = 0
            while (mag(pos - HOME_POS) < BASE_RADIUS * BASE_SCALE + MARGIN) and (tries < MINE_LOCATION_TRIES):
                pos[0] = np.random.random()
                pos[1] = np.random.random()
                tries += 1
            assert tries < MINE_LOCATION_TRIES
            self.mines.append(Mine(self.ore_cnt, *pos))
            if mine_distributions:
                self.mines[i].distributions = mine_distributions[i]

    def initialize_mines(self):
        """Assign a random rotation to each mine, and initialize the necessary sprites
        for the Pygame backend
        """

        for mine in self.mines:
            mine.rotation = np.random.randint(0, 360)

        self.mine_sprites = pygame.sprite.Group()
        self.mine_rects = []
        for mine in self.mines:
            mine_sprite = pygame.sprite.Sprite()
            # mine_sprite.image = pygame.transform.rotozoom(
            #    pygame.image.load(MINE_IMG), mine.rotation, MINE_SCALE,
            # ).convert_alpha()
            mine_sprite.image = pygame.image.load(MINE_IMG)
            mine_sprite.image = pygame.transform.scale(
                mine_sprite.image,
                (int(mine_sprite.image.get_width() * MINE_SCALE), int(mine_sprite.image.get_height() * MINE_SCALE)),
            )
            mine_sprite.image = pygame.transform.rotate(mine_sprite.image, mine.rotation)
            if self.render_mode == "human":
                mine_sprite.image = mine_sprite.image.convert_alpha()
            self.mine_sprites.add(mine_sprite)
            mine_sprite.rect = mine_sprite.image.get_rect()
            mine_sprite.rect.centerx = (mine.pos[0] * (1 - 2 * MARGIN)) * WIDTH + MARGIN * WIDTH
            mine_sprite.rect.centery = (mine.pos[1] * (1 - 2 * MARGIN)) * HEIGHT + MARGIN * HEIGHT
            self.mine_rects.append(mine_sprite.rect)

    def step(self, action):
        change = False  # Keep track of whether the state has changed
        reward = np.zeros(self.ore_cnt + 1, dtype=np.float32)

        reward[-1] = FUEL_IDLE * self.frame_skip

        if action == ACT_ACCEL:
            reward[-1] += FUEL_ACC * self.frame_skip
        elif action == ACT_MINE:
            reward[-1] += FUEL_MINE * self.frame_skip

        for _ in range(self.frame_skip if self.incremental_frame_skip else 1):
            if action == ACT_LEFT:
                self.cart.rotate(-ROTATION * (1 if self.incremental_frame_skip else self.frame_skip))
                change = True
            elif action == ACT_RIGHT:
                self.cart.rotate(ROTATION * (1 if self.incremental_frame_skip else self.frame_skip))
                change = True
            elif action == ACT_ACCEL:
                self.cart.accelerate(ACCELERATION * (1 if self.incremental_frame_skip else self.frame_skip))
            elif action == ACT_BRAKE:
                self.cart.accelerate(-DECELERATION * (1 if self.incremental_frame_skip else self.frame_skip))
            elif action == ACT_MINE:
                for _ in range(1 if self.incremental_frame_skip else self.frame_skip):
                    change = self.mine() or change

            if self.end:
                break

            for _ in range(1 if self.incremental_frame_skip else self.frame_skip):
                change = self.cart.step() or change

            distanceFromBase = mag(self.cart.pos - HOME_POS)
            if distanceFromBase < BASE_RADIUS * BASE_SCALE:
                if self.cart.departed:
                    # Cart left base then came back, ending the episode
                    self.end = True
                    # Sell resources
                    reward[: self.ore_cnt] += self.cart.content
                    self.cart.content = np.zeros(self.ore_cnt)
            else:
                # Cart left base
                self.cart.departed = True

        if change and self.image_observation:
            self.render_pygame()
        if self.render_mode == "human":
            self.render()

        return self.get_state(change), reward, self.end, False, {}

    def mine(self):
        """Perform the MINE action

        Returns:
            bool -- True if something was mined
        """
        if self.cart.speed < EPS_SPEED:
            # Get closest mine
            mine = min(self.mines, key=lambda mine: mine.distance(self.cart))

            if mine.mineable(self.cart):
                cart_free = self.capacity - np.sum(self.cart.content)
                mined = mine.mine()
                total_mined = np.sum(mined)
                if total_mined > cart_free:
                    # Scale mined content to remaining capacity
                    scale = cart_free / total_mined
                    mined = np.array(mined) * scale

                self.cart.content += mined

                if np.sum(mined) > 0:
                    return True
        return False

    def get_pixels(self, update=True):
        """Get the environment's image representation

        Keyword Arguments:
            update {bool} -- Whether to redraw the environment (default: {True})

        Returns:
            np.array -- array of pixels, with shape (width, height, channels)
        """
        if update:
            self.pixels = np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

        return self.pixels

    def get_state(self, update=True):
        """Returns the environment's state

        Keyword Arguments:
            update {bool} -- Whether to update the representation (default: {True})

        Returns:
            dict -- dict containing the aforementioned elements
        """
        if self.image_observation:
            state = self.get_pixels(update)
        else:
            angle = math.radians(self.cart.angle)
            sina = math.sin(angle)
            cosa = math.cos(angle)
            angle = np.array([sina, cosa], dtype=np.float32)
            state = np.concatenate(
                (
                    self.cart.pos,
                    np.array([self.cart.speed]),
                    angle,
                    self.cart.content / self.capacity,
                ),
                dtype=np.float32,
            )
        return state

        """ return {
            "position": self.cart.pos,
            "speed": self.cart.speed,
            "orientation": self.cart.angle,
            "content": self.cart.content,
            "pixels": self.get_pixels(update)
        } """

    def reset(self, seed=None, **kwargs):
        """Resets the environment to the start state

        Returns:
            [type] -- [description]
        """
        super().reset(seed=seed)

        if self.canvas is None and self.image_observation:
            self.render()  # init pygame

        if self.image_observation:
            self.render_pygame()

        self.cart.content = np.zeros(self.ore_cnt)
        self.cart.pos = np.array(HOME_POS)
        self.cart.speed = 0
        self.cart.angle = 45
        self.cart.departed = False
        self.end = False
        if self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def __str__(self):
        string = f"Completed: {self.end} "
        string += f"Departed: {self.cart.departed} "
        string += f"Content: {self.cart.content} "
        string += f"Speed: {self.cart.speed} "
        string += f"Direction: {self.cart.angle} ({self.cart.angle * math.pi / 180}) "
        string += f"Position: {self.cart.pos} "
        return string

    def render(self):
        if self.canvas is None or self.last_render_mode_used != self.render_mode:
            self.last_render_mode_used = self.render_mode
            pygame.init()
            self.canvas = pygame.Surface((WIDTH, HEIGHT))
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (WIDTH, HEIGHT),
                )

            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()

            self.initialize_mines()

            self.cart_sprite = pygame.sprite.Sprite()
            self.cart_sprites = pygame.sprite.Group()
            self.cart_sprites.add(self.cart_sprite)
            self.cart_image = pygame.image.load(CART_IMG)
            if self.render_mode == "human":
                self.cart_image = self.cart_image.convert_alpha()
            self.cart_image = pygame.transform.scale(
                self.cart_image,
                (int(self.cart_image.get_width() * CART_SCALE), int(self.cart_image.get_height() * CART_SCALE)),
            )

        if not self.image_observation:
            self.render_pygame()  # if the obs is not an image, then step would not have rendered the screen

        if self.render_mode == "human":
            self.screen.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(FPS)
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

    def render_pygame(self):
        self.mine_sprites.update()

        # Clear canvas
        self.canvas.fill(GRAY)

        # Draw Home
        pygame.draw.circle(
            self.canvas,
            RED,
            (int(WIDTH * HOME_X), int(HEIGHT * HOME_Y)),
            int(WIDTH / 3 * BASE_SCALE),
        )

        # Draw Mines
        self.mine_sprites.draw(self.canvas)

        # Draw cart
        self.cart_sprite.image = rot_center(self.cart_image, -self.cart.angle).copy()

        self.cart_sprite.rect = self.cart_sprite.image.get_rect(center=(200, 200))

        self.cart_sprite.rect.centerx = self.cart.pos[0] * (1 - 2 * MARGIN) * WIDTH + MARGIN * WIDTH
        self.cart_sprite.rect.centery = self.cart.pos[1] * (1 - 2 * MARGIN) * HEIGHT + MARGIN * HEIGHT

        self.cart_sprites.update()

        self.cart_sprites.draw(self.canvas)

        # Draw cart content
        width = self.cart_sprite.rect.width / (2 * self.ore_cnt)
        height = self.cart_sprite.rect.height / 3
        content_width = (width + 1) * self.ore_cnt
        offset = (self.cart_sprite.rect.width - content_width) / 2
        for i in range(self.ore_cnt):
            rect_height = height * self.cart.content[i] / self.capacity

            if rect_height >= 1:
                pygame.draw.rect(
                    self.canvas,
                    self.ore_colors[i],
                    (
                        self.cart_sprite.rect.left + offset + i * (width + 1),
                        self.cart_sprite.rect.top + offset * 1.5,
                        width,
                        rect_height,
                    ),
                )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


class Mine:
    """Class representing an individual Mine"""

    def __init__(self, ore_cnt, x, y):
        self.distributions = [scipy.stats.norm(np.random.random(), np.random.random()) for _ in range(ore_cnt)]
        self.pos = np.array((x, y))

    def distance(self, cart):
        return mag(cart.pos - self.pos)

    def mineable(self, cart):
        return self.distance(cart) <= MINE_RADIUS * MINE_SCALE * CART_SCALE

    def mine(self):
        """Generates collected resources according to the mine's random
        distribution

        Returns:
            list -- list of collected resources
        """
        return [max(0.0, dist.rvs()) for dist in self.distributions]

    def distribution_means(self):
        """
        Computes the mean of the truncated normal distributions
        """
        means = np.zeros(len(self.distributions))

        for i, dist in enumerate(self.distributions):
            mean, std = dist.args
            means[i] = truncated_mean(mean, std, 0, float("inf"))
            if np.isnan(means[i]):
                means[i] = 0
        return means


class Cart:
    """Class representing the actual minecart"""

    def __init__(self, ore_cnt):
        self.ore_cnt = ore_cnt
        self.pos = np.array([HOME_X, HOME_Y])
        self.speed = 0
        self.angle = 45
        self.content = np.zeros(self.ore_cnt)
        self.departed = False  # Keep track of whether the agent has left the base

    def accelerate(self, acceleration):
        self.speed = clip(self.speed + acceleration, 0, MAX_SPEED)

    def rotate(self, rotation):
        self.angle = (self.angle + rotation) % 360

    def step(self):
        """
        Update cart's position, taking the current speed into account
        Colliding with a border at anything but a straight angle will cause
        cart to "slide" along the wall.
        """
        pre = np.copy(self.pos)
        if self.speed < EPS_SPEED:
            return False
        x_velocity = self.speed * math.cos(self.angle * math.pi / 180)
        y_velocity = self.speed * math.sin(self.angle * math.pi / 180)
        x, y = self.pos
        if y != 0 and y != 1 and (y_velocity > 0 + EPS_SPEED or y_velocity < 0 - EPS_SPEED):
            if x == 1 and x_velocity > 0:
                self.angle += math.copysign(ROTATION, y_velocity)
            if x == 0 and x_velocity < 0:
                self.angle -= math.copysign(ROTATION, y_velocity)
        if x != 0 and x != 1 and (x_velocity > 0 + EPS_SPEED or x_velocity < 0 - EPS_SPEED):
            if y == 1 and y_velocity > 0:
                self.angle -= math.copysign(ROTATION, x_velocity)

            if y == 0 and y_velocity < 0:
                self.angle += math.copysign(ROTATION, x_velocity)

        self.pos[0] = clip(x + x_velocity, 0, 1)
        self.pos[1] = clip(y + y_velocity, 0, 1)
        self.speed = mag(pre - self.pos)

        return True


def compute_angle(p0, p1, p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


def rot_center(image, angle):
    """Rotate an image while preserving its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image


def mag(vector2d):
    return np.sqrt(np.dot(vector2d, vector2d))


def clip(val, lo, hi):
    return lo if val <= lo else hi if val >= hi else val


def scl(c):
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)


def truncated_mean(mean, std, a, b):
    if std == 0:
        return mean
    from scipy.stats import norm

    a = (a - mean) / std
    b = (b - mean) / std
    PHIB = norm.cdf(b)
    PHIA = norm.cdf(a)
    phib = norm.pdf(b)
    phia = norm.pdf(a)

    trunc_mean = mean + ((phia - phib) / (PHIB - PHIA)) * std
    return trunc_mean


def pareto_filter(costs, minimize=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    from https://stackoverflow.com/a/40239615
    """
    costs_copy = np.copy(costs) if minimize else -np.copy(costs)
    is_efficient = np.arange(costs_copy.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs_copy):
        nondominated_point_mask = np.any(costs_copy < costs_copy[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs_copy = costs_copy[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    return [costs[i] for i in is_efficient]


if __name__ == "__main__":
    env = Minecart(render_mode="human", image_observation=True)
    terminated = False
    env.reset()
    while True:
        env.render()
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        # print(str(env))
        if terminated:
            env.reset()
