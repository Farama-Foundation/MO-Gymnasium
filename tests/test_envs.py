import gym
import mo_gym
from gym.utils.env_checker import check_env
from mo_gym import LinearReward


def test_deep_sea_treasure():
    env = mo_gym.make('deep-sea-treasure-v0')
    env = LinearReward(env)
    check_env(env)

def test_fishwood():
    env = mo_gym.make('fishwood-v0')
    env = LinearReward(env)
    check_env(env)

def test_four_room():
    env = mo_gym.make('four-room-v0')
    env = LinearReward(env)
    check_env(env)

def test_minecart():
    env = mo_gym.make('minecart-v0')
    env = LinearReward(env)
    check_env(env)

def test_mountaincar():
    env = mo_gym.make('mo-mountaincar-v0')
    env = LinearReward(env)
    check_env(env)

def test_resource_gathering():
    env = mo_gym.make('resource-gathering-v0')
    env = LinearReward(env)
    check_env(env)

def test_fruit_tree():
    env = mo_gym.make('fruit-tree-v0')
    env = LinearReward(env)
    check_env(env)

def test_mario():
    env = mo_gym.make('mo-supermario-v0')
    env = LinearReward(env)
    check_env(env)

def test_reacher():
    env = mo_gym.make('mo-reacher-v0')
    env = LinearReward(env)
    check_env(env)

# TODO: failing because of highway_env original code
""" def test_highway():
    env = mo_gym.make('mo-highway-v0')
    env = LinearReward(env)
    check_env(env)

def test_highway_fast():
    env = mo_gym.make('mo-highway-fast-v0')
    env = LinearReward(env)
    check_env(env) """

def test_halfcheetah():
    env = mo_gym.make('mo-halfcheetah-v4')
    env = LinearReward(env)
    check_env(env)

def test_hopper():
    env = mo_gym.make('mo-hopper-v4')
    env = LinearReward(env)
    check_env(env)

def test_breakable_bottles():
    env = gym.make('breakable-bottles-v0')
    env = LinearReward(env)
    # skip because of bug in gym (see https://github.com/openai/gym/issues/2956)
    # check_env(env)
