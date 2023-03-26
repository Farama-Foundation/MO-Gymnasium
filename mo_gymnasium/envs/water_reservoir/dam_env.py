from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces.box import Box
from gymnasium.utils import EzPickle


class DamEnv(gym.Env, EzPickle):
    """
    ## Description
    A Water reservoir environment.
    The agent executes a continuous action, corresponding to the amount of water released by the dam.

    A. Castelletti, F. Pianosi and M. Restelli, "Tree-based Fitted Q-iteration for Multi-Objective Markov Decision problems,"
    The 2012 International Joint Conference on Neural Networks (IJCNN),
    Brisbane, QLD, Australia, 2012, pp. 1-8, doi: 10.1109/IJCNN.2012.6252759.

    ## Observation Space
    The observation is a float corresponding to the current level of the reservoir.

    ## Action Space
    The action is a float corresponding to the amount of water released by the dam.
    If normalized_action is True, the action is a float between 0 and 1 corresponding to the percentage of water released by the dam.

    ## Reward Space
    There are up to 4 rewards:
     - cost due to excess level wrt a flooding threshold (upstream)
     - deficit in the water supply wrt the water demand
     - deficit in hydroelectric supply wrt hydroelectric demand
     - cost due to excess level wrt a flooding threshold (downstream)
     By default, only the first two are used.

     ## Starting State
     The reservoir is initialized with a random level between 0 and 160.

     ## Arguments
        - render_mode: The render mode to use. Can be 'human', 'rgb_array' or 'ansi'.
        - time_limit: The maximum number of steps until the episode is truncated.
        - nO: The number of objectives to use. Can be 2, 3 or 4.
        - penalize: Whether to penalize the agent for selecting an action out of bounds.
        - normalized_action: Whether to normalize the action space as a percentage [0, 1].
        - initial_state: The initial state of the reservoir. If None, a random state is used.

     ## Credits
     Code from:
     [Mathieu Reymond](https://gitlab.ai.vub.ac.be/mreymond/dam).
     Ported from:
     [Simone Parisi](https://github.com/sparisi/mips).

     Sky background image from: Paulina Riva (https://opengameart.org/content/sky-background)
    """

    S = 1.0  # Reservoir surface
    W_IRR = 50.0  # Water demand
    H_FLO_U = 50.0  # Flooding threshold (upstream, i.e. height of dam)
    S_MIN_REL = 100.0  # Release threshold (i.e. max capacity)
    DAM_INFLOW_MEAN = 40.0  # Random inflow (e.g. rain)
    DAM_INFLOW_STD = 10.0
    Q_MEF = 0.0
    GAMMA_H2O = 1000.0  # water density
    W_HYD = 4.36  # Hydroelectric demand
    Q_FLO_D = 30.0  # Flooding threshold (downstream, i.e. releasing too much water)
    ETA = 1.0  # Turbine efficiency
    G = 9.81  # Gravity

    utopia = {2: [-0.5, -9], 3: [-0.5, -9, -0.0001], 4: [-0.5, -9, -0.001, -9]}
    antiutopia = {2: [-2.5, -11], 3: [-65, -12, -0.7], 4: [-65, -12, -0.7, -12]}

    # Create colors.
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    s_init = np.array(
        [
            9.6855361e01,
            5.8046026e01,
            1.1615767e02,
            2.0164311e01,
            7.9191000e01,
            1.4013098e02,
            1.3101816e02,
            4.4351321e01,
            1.3185943e01,
            7.3508622e01,
        ],
        dtype=np.float32,
    )

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 2}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        time_limit: int = 100,
        nO=2,
        penalize: bool = False,
        normalized_action: bool = False,
        initial_state: Optional[np.ndarray] = None,
    ):
        EzPickle.__init__(self, render_mode, time_limit, nO, penalize, normalized_action)
        self.render_mode = render_mode

        self.observation_space = Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        self.normalized_action = normalized_action
        if self.normalized_action:
            self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)

        self.nO = nO
        self.penalize = penalize
        self.time_limit = time_limit
        self.initial_state = initial_state
        self.time_step = 0
        self.last_action = None
        self.dam_inflow = None
        self.excess = None
        self.defict = None

        low = -np.ones(nO) * np.inf  # DamEnv.antiutopia[nO]
        high = np.zeros(nO)  # DamEnv.utopia[nO]
        self.reward_space = Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.reward_dim = nO

        self.window = None
        self.window_size = (300, 200)  # width x height
        self.clock = None
        self.water_img = None
        self.wall_img = None
        self.sky_img = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.time_step = 0
        if self.initial_state is not None:
            state = self.initial_state
        else:
            if not self.penalize:
                state = self.np_random.choice(DamEnv.s_init, size=1)
            else:
                state = self.np_random.integers(0, 160, size=1)

        self.state = np.array(state, dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return self.state, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. mo_gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, render_mode: str):
        if self.window is None:
            pygame.init()

            if render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Water Reservoir")
                self.window = pygame.display.set_mode(self.window_size)
            else:
                self.window = pygame.Surface(self.window_size)

            if self.clock is None:
                self.clock = pygame.time.Clock()

            if self.water_img is None:
                self.water_img = pygame.image.load(path.join(path.dirname(__file__), "assets/water.png"))
            if self.wall_img is None:
                self.wall_img = pygame.image.load(path.join(path.dirname(__file__), "assets/wall.png"))
            if self.sky_img is None:
                self.sky_img = pygame.image.load(path.join(path.dirname(__file__), "assets/sky.png"))
                self.sky_img = pygame.transform.flip(self.sky_img, False, True)
                self.sky_img = pygame.transform.scale(self.sky_img, self.window_size)

            self.font = pygame.font.Font(path.join(path.dirname(__file__), "assets", "Minecraft.ttf"), 15)

        self.window.blit(self.sky_img, (0, 0))

        # Draw the dam
        for x in range(self.wall_img.get_width(), self.window_size[0] - self.wall_img.get_width(), self.water_img.get_width()):
            for y in range(self.window_size[1] - int(self.state[0]), self.window_size[1], self.water_img.get_height()):
                self.window.blit(self.water_img, (x, y))

        # Draw the wall
        for y in range(0, int(DamEnv.H_FLO_U), self.wall_img.get_width()):
            self.window.blit(self.wall_img, (0, self.window_size[1] - y - self.wall_img.get_height()))
            self.window.blit(
                self.wall_img,
                (self.window_size[0] - self.wall_img.get_width(), self.window_size[1] - y - self.wall_img.get_height()),
            )

        if self.last_action is not None:
            img = self.font.render(f"Water Released: {self.last_action:.2f}", True, (0, 0, 0))
            self.window.blit(img, (20, 10))
            img = self.font.render(f"Dam Inflow: {self.dam_inflow:.2f}", True, (0, 0, 0))
            self.window.blit(img, (20, 25))
            img = self.font.render(f"Water Level: {self.state[0]:.2f}", True, (0, 0, 0))
            self.window.blit(img, (20, 40))
            img = self.font.render(f"Demand Deficit: {self.defict:.2f}", True, (0, 0, 0))
            self.window.blit(img, (20, 55))
            img = self.font.render(f"Flooding Excess: {self.excess:.2f}", True, (0, 0, 0))
            self.window.blit(img, (20, 70))

        img = self.font.render("Flooding threshold", True, (255, 0, 0))
        self.window.blit(img, (20, self.window_size[1] - DamEnv.H_FLO_U))
        pygame.draw.line(
            self.window,
            (255, 0, 0),
            (0, self.window_size[1] - DamEnv.H_FLO_U),
            (self.window_size[0], self.window_size[1] - DamEnv.H_FLO_U),
        )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def _render_text(self):
        outfile = StringIO()
        outfile.write(f"Water level: {self.state[0]:.2f}\n")
        if self.last_action is not None:
            outfile.write(f"Water released: {self.last_action:.2f}\n")
            outfile.write(f"Dam inflow: {self.dam_inflow:.2f}\n")
            outfile.write(f"Demand deficit: {self.defict:.2f}\n")
            outfile.write(f"Flooding excess: {self.excess:.2f}\n")

        with closing(outfile):
            return outfile.getvalue()

    def step(self, action):
        # bound the action
        actionLB = np.clip(self.state - DamEnv.S_MIN_REL, 0, None)
        actionUB = self.state

        if self.normalized_action:
            action = action * (actionUB - actionLB) + actionLB
            penalty = 0.0
        else:
            # Penalty proportional to the violation
            bounded_action = np.clip(action, actionLB, actionUB)
            penalty = -self.penalize * np.abs(bounded_action - action)
            action = bounded_action

        # transition dynamic
        self.last_action = action[0]
        self.dam_inflow = self.np_random.normal(DamEnv.DAM_INFLOW_MEAN, DamEnv.DAM_INFLOW_STD, len(self.state))[0]
        # small chance dam_inflow < 0
        n_state = np.clip(self.state + self.dam_inflow - action, 0, None).astype(np.float32)

        # cost due to excess level wrt a flooding threshold (upstream)
        self.excess = np.clip(n_state / DamEnv.S - DamEnv.H_FLO_U, 0, None)[0]
        r0 = -self.excess + penalty
        # deficit in the water supply wrt the water demand
        self.defict = -np.clip(DamEnv.W_IRR - action, 0, None)[0]
        r1 = self.defict + penalty

        q = np.clip(action[0] - DamEnv.Q_MEF, 0, None)
        p_hyd = DamEnv.ETA * DamEnv.G * DamEnv.GAMMA_H2O * n_state[0] / DamEnv.S * q / 3.6e6

        # deficit in hydroelectric supply wrt hydroelectric demand
        r2 = -np.clip(DamEnv.W_HYD - p_hyd, 0, None) + penalty
        # cost due to excess level wrt a flooding threshold (downstream)
        r3 = -np.clip(action[0] - DamEnv.Q_FLO_D, 0, None) + penalty

        reward = np.array([r0, r1, r2, r3], dtype=np.float32)[: self.nO].flatten()

        self.state = n_state

        self.time_step += 1
        truncated = self.time_step >= self.time_limit
        terminated = False

        if self.render_mode == "human":
            self.render()

        return n_state, reward, terminated, truncated, {}


if __name__ == "__main__":
    import mo_gymnasium as mo_gym

    env = mo_gym.make("water-reservoir-v0", render_mode="human")
    obs, info = env.reset()
    while True:
        action = env.state
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
