import gym
from gym import spaces
import numpy as np
from pybulletgym.envs.roboschool.robots.robot_bases import MJCFBasedRobot
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene

target_positions = list(map(lambda l: np.array(l), [(0.14, 0.0), (-0.14, 0.0), (0.0, 0.14), (0.0, -0.14)]))


class ReacherBulletEnv(BaseBulletEnv):

    def __init__(self, target=(0.14, 0.0), fixed_initial_state=False):
        self.robot = ReacherRobot(target, fixed_initial_state=fixed_initial_state)
        BaseBulletEnv.__init__(self, self.robot)
        self._cam_dist = 0.75

        #self.target_positions = list(map(lambda l: np.array(l), [(0.14, 0.0), (-0.14, 0.0), (0.0, 0.14), (0.0, -0.14), (0.22, 0.0), (-0.22, 0.0), (0.0, 0.22), (0.0, -0.22), (0.1, 0.1), (0.1, -0.1), (-0.1, 0.1), (-0.1, -0.1)]))
        #self.target_positions = list(map(lambda l: np.array(l), [(0.14, 0.0), (-0.14, 0.0), (0.0, 0.14), (0.0, -0.14), (0.1, 0.1), (0.1, -0.1), (-0.1, 0.1), (-0.1, -0.1)]))
        self.target_positions = list(map(lambda l: np.array(l), [(0.14, 0.0), (-0.14, 0.0), (0.0, 0.14), (0.0, -0.14)]))

        actions = [-1., 0., 1.]
        self.action_dict = dict()
        for a1 in actions:
            for a2 in actions:
                self.action_dict[len(self.action_dict)] = (a1, a2)
        
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def step(self, a):
        real_action = self.action_dict[a]

        assert (not self.scene.multiplayer)
        self.robot.apply_action(real_action)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.to_target_vec
        
        """ delta = np.linalg.norm(np.array(self.robot.fingertip.pose().xyz()) - np.array(self.robot.target.pose().xyz()))
        reward = 1. - 4. * delta """

        phi = np.zeros(len(self.target_positions))
        for index, target in enumerate(self.target_positions):
            delta = np.linalg.norm(np.array(self.robot.fingertip.pose().xyz()[:2]) - target)
            phi[index] = (1. - 4*delta) # 1 - 4

        self.HUD(state, real_action, False)
        
        return state, phi, False, False, {}

    def camera_adjust(self):
        x, y, z = self.robot.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
    
    def reset(self, seed=None, **kwargs):
        self._seed(seed)
        return super().reset(), {}


class ReacherRobot(MJCFBasedRobot):
    TARG_LIMIT = 0.27

    def __init__(self, target, fixed_initial_state=False):
        MJCFBasedRobot.__init__(self, 'reacher.xml', 'body0', action_dim=2, obs_dim=4)
        self.target_pos = target
        self.fixed_initial_state = fixed_initial_state

    def robot_specific_reset(self, bullet_client):
        self.jdict["target_x"].reset_current_position(target_positions[0][0], 0)
        self.jdict["target_y"].reset_current_position(target_positions[0][1], 0)

        """ self.jdict["target2_x"].reset_current_position(target_positions[1][0], 0)
        self.jdict["target2_y"].reset_current_position(target_positions[1][1], 0)
        self.jdict["target3_x"].reset_current_position(target_positions[2][0], 0)
        self.jdict["target3_y"].reset_current_position(target_positions[2][1], 0)
        self.jdict["target4_x"].reset_current_position(target_positions[3][0], 0)
        self.jdict["target4_y"].reset_current_position(target_positions[3][1], 0) """

        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]
        self.central_joint = self.jdict["joint0"]
        self.elbow_joint = self.jdict["joint1"]
        if not self.fixed_initial_state:
            self.central_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
            self.elbow_joint.reset_current_position(self.np_random.uniform(low=-3.14 / 2, high=3.14 / 2), 0)
        else:
            self.central_joint.reset_current_position(0, 0)
            self.elbow_joint.reset_current_position(3.1415/2, 0)

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        self.central_joint.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
        self.elbow_joint.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))

    def calc_state(self):
        theta, self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        # target_x, _ = self.jdict["target_x"].current_position()
        # target_y, _ = self.jdict["target_y"].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return np.array([
            np.cos(theta),
            np.sin(theta),
            self.theta_dot*0.1,
            self.gamma,
            self.gamma_dot*0.1,
        ], dtype=np.float32)


if __name__ == '__main__':

    env = ReacherBulletEnv()
    #env.render(mode='human')
    obs = env.reset()
    print(env.observation_space.contains(obs), obs.dtype, env.observation_space)
    while True:
        env.step(env.action_space.sample())
        #env.render(mode='human')