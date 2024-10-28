from os import path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.reacher_v4 import ReacherEnv
from gymnasium.spaces import Box, Discrete


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class MOReacherEnv(ReacherEnv):
    """
    ## Description
    Multi-objective version of the [`Reacher-v4` environment](https://gymnasium.farama.org/environments/mujoco/reacher/).

    ## Observation Space
    The observation is 6-dimensional and contains:
    - sin and cos of the angles of the central and elbow joints
    - angular velocity of the central and elbow joints

    ## Action Space
    The action space is discrete and contains the 3^2=9 possible actions based on applying positive (+1), negative (-1) or zero (0) torque to each of the two joints.

    ## Reward Space
    The reward is 4-dimensional and is defined based on the distance of the tip of the arm and the four target locations.
    For each i={1,2,3,4} it is computed as:
    ```math
        r_i = 1  - 4 * || finger_tip_coord - target_i ||^2
    ```
    """

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            path.join(path.dirname(__file__), "assets", "mo_reacher.xml"),
            2,
            observation_space=self.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        actions = [-1.0, 0.0, 1.0]
        self.action_dict = dict()
        for a1 in actions:
            for a2 in actions:
                self.action_dict[len(self.action_dict)] = (a1, a2)
        self.action_space = Discrete(9)
        # Target goals: x1, y1, x2, y2, ... x4, y4
        self.goal = np.array([0.14, 0.0, -0.14, 0.0, 0.0, 0.14, 0.0, -0.14])
        self.reward_space = Box(low=-1.0, high=1.0, shape=(4,))
        self.reward_dim = 4

    def step(self, a):
        real_action = self.action_dict[int(a)]
        vec_reward = np.array(
            [
                1 - 4 * np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target1")[:2]),
                1 - 4 * np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target2")[:2]),
                1 - 4 * np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target3")[:2]),
                1 - 4 * np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target4")[:2]),
            ],
            dtype=np.float32,
        )

        self._step_mujoco_simulation(real_action, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        return (
            ob,
            vec_reward,
            False,
            False,
            {},
        )

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[:2] = np.array([0, 3.1415 / 2])  # init position
        qpos[-len(self.goal) :] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qvel.flat[:2] * 0.1,
            ]
        )
