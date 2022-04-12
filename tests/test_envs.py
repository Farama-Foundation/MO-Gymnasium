import gym
import mo_gym
from gym.utils.env_checker import check_env
from mo_gym import LinearReward


def test_deep_sea_treasure():
    env = gym.make('deep-sea-treasure-v0')
    env = LinearReward(env)
    check_env(env)

def test_four_room():
    env = gym.make('four-room-v0')
    env = LinearReward(env)
    check_env(env)

def test_minecart():
    env = gym.make('minecart-v0')
    env = LinearReward(env)
    check_env(env)
