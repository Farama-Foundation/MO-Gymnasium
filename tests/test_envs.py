import pickle

import gymnasium as gym
from gymnasium.utils.env_checker import check_env, data_equivalence

import mo_gymnasium as mo_gym
from mo_gymnasium import LinearReward


def _test_pickle_env(env: gym.Env):
    pickled_env = pickle.loads(pickle.dumps(env))

    data_equivalence(env.reset(), pickled_env.reset())

    action = env.action_space.sample()
    data_equivalence(env.step(action), pickled_env.step(action))
    env.close()
    pickled_env.close()


def test_deep_sea_treasure():
    env = mo_gym.make("deep-sea-treasure-v0")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_fishwood():
    env = mo_gym.make("fishwood-v0")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_four_room():
    env = mo_gym.make("four-room-v0")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_minecart():
    env = mo_gym.make("minecart-v0")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_mountaincar():
    env = mo_gym.make("mo-mountaincar-v0")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_continuous_mountaincar():
    env = mo_gym.make("mo-mountaincarcontinuous-v0")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_resource_gathering():
    env = mo_gym.make("resource-gathering-v0")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_fruit_tree():
    env = mo_gym.make("fruit-tree-v0")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_mario():
    env = mo_gym.make("mo-supermario-v0")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_reacher_pybullet():
    env = mo_gym.make("mo-reacher-v0")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_reacher_mujoco():
    env = mo_gym.make("mo-reacher-v4")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


# TODO: failing because highway_env is not deterministic given a random seed
""" def test_highway():
    env = mo_gym.make('mo-highway-v0')
    env = LinearReward(env)
    check_env(env)

def test_highway_fast():
    env = mo_gym.make('mo-highway-fast-v0')
    env = LinearReward(env)
    check_env(env) """


def test_halfcheetah():
    env = mo_gym.make("mo-halfcheetah-v4")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_hopper():
    env = mo_gym.make("mo-hopper-v4")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_breakable_bottles():
    env = mo_gym.make("breakable-bottles-v0")
    env = LinearReward(env)
    check_env(env)
    _test_pickle_env(env)


def test_water_reservoir():
    env = mo_gym.make("water-reservoir-v0")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


def test_lunar_lander():
    env = mo_gym.make("mo-lunar-lander-v2")
    env = LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)
