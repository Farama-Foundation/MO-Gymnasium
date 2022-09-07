import numpy as np
import gym
#from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gym.wrappers import (FrameStack, GrayScaleObservation, ResizeObservation, TimeLimit)
from gym.utils import seeding
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros import SuperMarioBrosEnv
import gym_super_mario_bros
#import matplotlib.pyplot as plt

import mo_gym


class MOSuperMarioBros(SuperMarioBrosEnv):
    
    def __init__(self, rom_mode='pixel', lost_levels=False, target=None, objectives=['x_pos', 'time', 'death', 'coin', 'enemy']):
        super().__init__(rom_mode, lost_levels, target)

        self.objectives = set(objectives)
        self.reward_space = gym.spaces.Box(high=np.inf, low=-np.inf, shape=(len(objectives),))

        self.single_stage = True
        self.done_when_dead = True

    def reset(self, seed=None, **kwargs):
        self._np_random, seed = seeding.np_random(seed) # this is not used
        self.coin = 0
        self.x_pos = 0
        self.time = 0
        self.score = 0
        self.stage_bonus = 0
        self.lives = 2
        return super().reset(), {}

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.single_stage and info["flag_get"]:
            self.stage_bonus = 10000
            done = True

        ''' Construct Multi-Objective Reward'''
        # [x_pos, time, death, coin, enemy]
        moreward = []

        # 1. x position
        if 'x_pos' in self.objectives:
            xpos_r = info["x_pos"] - self.x_pos
            self.x_pos = info["x_pos"]
            # resolve an issue where after death the x position resets
            if xpos_r < -5:
                xpos_r = 0
            moreward.append(xpos_r)
        
        # 2. time penaltiy 
        if 'time' in self.objectives:
            time_r = info["time"] - self.time
            self.time = info["time"]
            # time is aways decreasing
            if time_r > 0:
                time_r = 0
            moreward.append(time_r)

        # 3. death
        if 'death' in self.objectives:
            if self.lives > info['life']:
                death_r = -25
            else:
                death_r = 0
            moreward.append(death_r)

        # 4. coin
        if 'coin' in self.objectives:
            coin_r = (info['coins'] - self.coin) * 100
            self.coin = info['coins']
            moreward.append(coin_r)

        # 5. enemy
        if 'enemy' in self.objectives:
            enemy_r = info['score'] - self.score
            if coin_r > 0 or done:
                enemy_r = 0
            self.score = info['score']
            moreward.append(enemy_r)

        ############################################################################

        if self.done_when_dead:
            # when Mario loses life, changes the state to the terminal
            if self.lives > info['life'] and info['life'] > 0:
                done = True

        self.lives = info['life']
       
        mor = np.array(moreward, dtype=np.float32) * self.reward_space.shape[0] / 150

        info['score'] = info['score'] + self.stage_bonus

        return obs, mor, bool(done), False, info


if __name__ == '__main__':

    env = MOSuperMarioBros()
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    #env = MaxAndSkipEnv(env, 4)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)
    #env = FrameStack(env, 4)
    env = mo_gym.LinearReward(env)

    done = False
    env.reset()
    while True:
        obs, r, done, info = env.step(env.action_space.sample())
        print(r, info['vector_reward'], done, info['time'])
        """ plt.figure()
        plt.imshow(obs, cmap='gray', vmin=0, vmax=255)
        plt.show() """
        env.render()
        if done:
            env.reset()
