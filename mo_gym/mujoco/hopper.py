import numpy as np
from gym.envs.mujoco.hopper_v4 import HopperEnv
from gym.spaces import Box


class MOHopperEnv(HopperEnv):
    def __init__(self, cost_objective=True, **kwargs):
        super(MOHopperEnv, self).__init__(**kwargs)
        self.cost_objetive = cost_objective
        self.rew_dim = 3 if cost_objective else 2
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.rew_dim,))

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        # ctrl_cost = self.control_cost(action)

        # forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        # rewards = forward_reward + healthy_reward
        # costs = ctrl_cost

        observation = self._get_obs()
        # reward = rewards - costs
        terminated = self.terminated

        z = self.data.qpos[1]
        height = 10 * (z - self.init_qpos[1])
        energy_cost = np.sum(np.square(action))

        if self.cost_objetive:
            vec_reward = np.array([x_velocity, height, -energy_cost], dtype=np.float32)
        else:
            vec_reward = np.array([x_velocity, height], dtype=np.float32)
            vec_reward -= self._ctrl_cost_weight * energy_cost

        vec_reward += healthy_reward

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "height_reward": height,
            "energy_reward": -energy_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, vec_reward, terminated, False, info
