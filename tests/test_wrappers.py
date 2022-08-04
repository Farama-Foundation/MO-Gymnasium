import numpy as np

import mo_gym
from mo_gym import MONormalizeReward, MOClipReward, MOSyncVectorEnv


def go_to_8_3(env):
    """
    Goes to (8.2, -3) treasure, returns the rewards
    """
    env.reset()
    env.step(3)  # right
    env.step(1)  # down
    _, rewards, _, _ = env.step(1)
    return rewards


def test_normalization_wrapper():
    env = mo_gym.make('deep-sea-treasure-v0')
    norm_treasure_env = MONormalizeReward(env, idx=0)
    both_norm_env = MONormalizeReward(norm_treasure_env, idx=1)

    # Tests for both rewards normalized
    for i in range(30):
        go_to_8_3(both_norm_env)
    both_norm_env.reset()
    _, rewards, _, _ = both_norm_env.step(1)  # down
    np.testing.assert_allclose(rewards, [0.18, -1.24], rtol=0, atol=1e-2)
    rewards = go_to_8_3(both_norm_env)
    np.testing.assert_allclose(rewards, [2.13, -1.24], rtol=0, atol=1e-2)

    # Tests for only treasure normalized
    for i in range(30):
        go_to_8_3(norm_treasure_env)
    norm_treasure_env.reset()
    _, rewards, _, _ = norm_treasure_env.step(1)  # down
    # Time rewards are not normalized (-1)
    np.testing.assert_allclose(rewards, [0.18, -1.], rtol=0, atol=1e-2)
    rewards = go_to_8_3(norm_treasure_env)
    np.testing.assert_allclose(rewards, [2.13, -1.], rtol=0, atol=1e-2)


def test_clip_wrapper():
    env = mo_gym.make('deep-sea-treasure-v0')
    clip_treasure_env = MOClipReward(env, idx=0, min_r=0, max_r=0.5)
    both_clipped_env = MOClipReward(clip_treasure_env, idx=1, min_r=-0.5, max_r=0)

    # Tests for both rewards clipped
    both_clipped_env.reset()
    _, rewards, _, _ = both_clipped_env.step(1)  # down
    np.testing.assert_allclose(rewards, [0.5, -0.5], rtol=0, atol=1e-2)
    rewards = go_to_8_3(both_clipped_env)
    np.testing.assert_allclose(rewards, [0.5, -0.5], rtol=0, atol=1e-2)

    # Tests for only treasure clipped
    clip_treasure_env.reset()
    _, rewards, _, _ = clip_treasure_env.step(1)  # down
    # Time rewards are not clipped (-1)
    np.testing.assert_allclose(rewards, [0.5, -1.], rtol=0, atol=1e-2)
    rewards = go_to_8_3(clip_treasure_env)
    np.testing.assert_allclose(rewards, [0.5, -1.], rtol=0, atol=1e-2)


def test_mo_sync_wrapper():
    def make_env(env_id):
        def thunk():
            env = mo_gym.make(env_id)
            return env

        return thunk

    num_envs = 3
    envs = MOSyncVectorEnv([
        make_env("deep-sea-treasure-v0") for _ in range(num_envs)
    ])

    envs.reset()
    obs, rewards, dones, infos = envs.step(envs.action_space.sample())
    assert len(obs) == num_envs, "Number of observations do not match the number of envs"
    assert len(rewards) == num_envs, "Number of rewards do not match the number of envs"
    assert len(dones) == num_envs, "Number of dones do not match the number of envs"
