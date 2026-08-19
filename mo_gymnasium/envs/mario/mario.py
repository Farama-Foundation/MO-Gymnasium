from typing import Optional

import gymnasium as gym
import numpy as np
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# from nes_py.wrappers import JoypadSpace
from mo_gymnasium.envs.mario.joypad_space import JoypadSpace


class MOSuperMarioBros(SuperMarioBrosEnv):
    """
    ## Description
    Multi-objective version of the SuperMarioBro environment.

    Obs: To run this environment, it is required numpy==1.21 due to nes-py dependency.

    See [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) for more information.

    ## Reward Space
    The reward is a 5-dimensional vector:
    - 0: How far Mario moved in the x position
    - 1: Time penalty for how much time has passed between two time steps
    - 2: -25 if Mario died, 0 otherwise
    - 3: +100 if Mario collected coins, else 0
    - 4: Points for killing an enemy

    ## Episode Termination
    The episode terminates when Mario dies or reaches the flag.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        lost_levels=False,
        target=None,
        objectives=["x_pos", "time", "death", "coin", "enemy"],
        death_as_penalty=False,
        render_mode: Optional[str] = None,
    ):
        self.render_mode = render_mode
        super().__init__(lost_levels, target, render_mode=render_mode)

        self.objectives = set(objectives)
        self.death_as_penalty = death_as_penalty
        if self.death_as_penalty:  # death is not a separate objective
            self.objectives.discard("death")
        self.reward_dim = len(self.objectives)

        low = np.empty(self.reward_dim, dtype=np.float32)
        high = np.empty(self.reward_dim, dtype=np.float32)
        obj_idx = 0
        if "x_pos" in self.objectives:
            low[obj_idx] = -np.inf
            high[obj_idx] = np.inf
            obj_idx += 1
        if "time" in self.objectives:
            low[obj_idx] = -np.inf
            high[obj_idx] = 0.0
            obj_idx += 1
        if "death" in self.objectives:
            low[obj_idx] = -25.0
            high[obj_idx] = 0.0
            obj_idx += 1
        if "coin" in self.objectives:
            low[obj_idx] = 0.0
            high[obj_idx] = 100.0
            obj_idx += 1
        if "enemy" in self.objectives:
            low[obj_idx] = 0.0
            high[obj_idx] = np.inf

        self.reward_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(len(self.objectives),),
        )

        self.score = 0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.score = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        reward_components = info["reward_components"]
        # breakpoint()

        """ Construct Multi-Objective Reward"""
        # [x_pos, time, death, coin, enemy]
        vec_reward = np.zeros(self.reward_dim, dtype=np.float32)
        obj_idx = 0

        # 1. x position
        if "x_pos" in self.objectives:
            vec_reward[obj_idx] = reward_components["progress"]
            obj_idx += 1

        # 2. time penaltiy
        if "time" in self.objectives:
            vec_reward[obj_idx] = reward_components["time"]
            obj_idx += 1

        # 3. death
        if "death" in self.objectives:
            vec_reward[obj_idx] = reward_components["death"]
            obj_idx += 1

        # 4. coin
        coin_r = 0.0
        if "coin" in self.objectives:
            coin_r = reward_components["coins"]
            vec_reward[obj_idx] = coin_r
            obj_idx += 1

        # 5. enemy
        if "enemy" in self.objectives:
            enemy_r = info["score"] - self.score
            if coin_r > 0 or terminated:
                enemy_r = 0
            self.score = info["score"]
            vec_reward[obj_idx] = enemy_r
            obj_idx += 1

        if self.death_as_penalty:
            vec_reward += reward_components["death"]  # add death reward to all objectives

        vec_reward *= self.reward_space.shape[0] / 150

        return obs, vec_reward, terminated, truncated, info


if __name__ == "__main__":
    from gymnasium.wrappers import ResizeObservation
    from gymnasium.wrappers.transform_observation import GrayscaleObservation

    import mo_gymnasium as mo_gym

    env = mo_gym.make("mo-supermario-v1", render_mode="human", objectives=["x_pos", "time", "death", "coin", "enemy"])
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # env = MaxAndSkipEnv(env, 4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    # env = FrameStack(env, 4)
    # env = mo_gym.wrappers.LinearReward(env)

    terminated = False
    env.reset()
    return_vect = np.zeros(env.unwrapped.reward_dim, dtype=np.float32)
    while True:
        action = env.action_space.sample()  # int(input("Enter action (0-6): "))
        obs, r, terminated, truncated, info = env.step(action)
        return_vect += r
        print(r, terminated)
        # plt.figure()
        # plt.imshow(obs, cmap='gray', vmin=0, vmax=255)
        # plt.show()
        env.render()
        if r[-2] != 0 or r[-1] != 0:
            input()
        if terminated or truncated:
            input("Press Enter to continue...")
            print("Episode return:", return_vect)
            exit()
            env.reset()
