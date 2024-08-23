import numpy as np

import mo_gymnasium as mo_gym
from mo_gymnasium import (
    MOClipReward,
    MONormalizeReward,
    MORecordEpisodeStatistics,
    MOSyncVectorEnv,
)


def go_to_8_3(env):
    """
    Goes to (8.2, -3) treasure, returns the rewards
    """
    env.reset()
    env.step(3)  # action: right, rewards: [0, -1]
    env.step(1)  # action: down, rewards: [0, -1]
    _, rewards, _, _, infos = env.step(1)  # action: down, rewards: [8.2, -1]
    return rewards, infos


def test_normalization_wrapper():
    env = mo_gym.make("deep-sea-treasure-v0")
    norm_treasure_env = MONormalizeReward(env, idx=0)
    both_norm_env = MONormalizeReward(norm_treasure_env, idx=1)

    # Tests for both rewards normalized
    for i in range(30):
        go_to_8_3(both_norm_env)
    both_norm_env.reset()
    _, rewards, _, _, _ = both_norm_env.step(1)  # down
    np.testing.assert_allclose(rewards, [0.18, -1.24], rtol=0, atol=1e-2)
    rewards, _ = go_to_8_3(both_norm_env)
    np.testing.assert_allclose(rewards, [2.13, -1.24], rtol=0, atol=1e-2)

    # Tests for only treasure normalized
    for i in range(30):
        go_to_8_3(norm_treasure_env)
    norm_treasure_env.reset()
    _, rewards, _, _, _ = norm_treasure_env.step(1)  # down
    # Time rewards are not normalized (-1)
    np.testing.assert_allclose(rewards, [0.18, -1.0], rtol=0, atol=1e-2)
    rewards, _ = go_to_8_3(norm_treasure_env)
    np.testing.assert_allclose(rewards, [2.13, -1.0], rtol=0, atol=1e-2)


def test_clip_wrapper():
    env = mo_gym.make("deep-sea-treasure-v0")
    clip_treasure_env = MOClipReward(env, idx=0, min_r=0, max_r=0.5)
    both_clipped_env = MOClipReward(clip_treasure_env, idx=1, min_r=-0.5, max_r=0)

    # Tests for both rewards clipped
    both_clipped_env.reset()
    _, rewards, _, _, _ = both_clipped_env.step(1)  # down
    np.testing.assert_allclose(rewards, [0.5, -0.5], rtol=0, atol=1e-2)
    rewards, _ = go_to_8_3(both_clipped_env)
    np.testing.assert_allclose(rewards, [0.5, -0.5], rtol=0, atol=1e-2)

    # Tests for only treasure clipped
    clip_treasure_env.reset()
    _, rewards, _, _, _ = clip_treasure_env.step(1)  # down
    # Time rewards are not clipped (-1)
    np.testing.assert_allclose(rewards, [0.5, -1.0], rtol=0, atol=1e-2)
    rewards, _ = go_to_8_3(clip_treasure_env)
    np.testing.assert_allclose(rewards, [0.5, -1.0], rtol=0, atol=1e-2)


def test_mo_sync_wrapper():
    def make_env(env_id):
        def thunk():
            env = mo_gym.make(env_id)
            env = MORecordEpisodeStatistics(env, gamma=0.97)
            return env

        return thunk

    num_envs = 3
    envs = MOSyncVectorEnv([make_env("deep-sea-treasure-v0") for _ in range(num_envs)])

    envs.reset()
    obs, rewards, terminateds, truncateds, infos = envs.step(envs.action_space.sample())
    assert len(obs) == num_envs, "Number of observations do not match the number of envs"
    assert len(rewards) == num_envs, "Number of rewards do not match the number of envs"
    assert len(terminateds) == num_envs, "Number of terminateds do not match the number of envs"
    assert len(truncateds) == num_envs, "Number of truncateds do not match the number of envs"


def test_mo_record_ep_statistic():
    env = mo_gym.make("deep-sea-treasure-v0")
    env = MORecordEpisodeStatistics(env, gamma=0.97)

    env.reset()
    _, info = go_to_8_3(env)

    assert isinstance(info["episode"]["r"], np.ndarray)
    assert isinstance(info["episode"]["dr"], np.ndarray)
    assert info["episode"]["r"].shape == (2,)
    assert info["episode"]["dr"].shape == (2,)
    assert tuple(info["episode"]["r"]) == (np.float32(8.2), np.float32(-3.0))
    np.testing.assert_allclose(info["episode"]["dr"], [7.71538, -2.9109], rtol=0, atol=1e-2)
    # 0 * 0.97**0 + 0 * 0.97**1 + 8.2 * 0.97**2 == 7.71538
    # -1 * 0.97**0 + -1 * 0.97**1 + -1 * 0.97**2 == -2.9109
    assert isinstance(info["episode"]["l"], np.int32)
    assert info["episode"]["l"] == 3
    assert isinstance(info["episode"]["t"], np.float32)


def test_mo_record_ep_statistic_vector_env():
    def make_env(env_id):
        def thunk():
            env = mo_gym.make(env_id)
            return env

        return thunk

    num_envs = 3
    envs = MOSyncVectorEnv([make_env("deep-sea-treasure-v0") for _ in range(num_envs)])
    envs = MORecordEpisodeStatistics(envs)

    envs.reset()
    terminateds = np.array([False] * num_envs)
    info = {}
    while not np.any(terminateds):
        obs, rewards, terminateds, _, info = envs.step(envs.action_space.sample())

    assert isinstance(info["episode"]["r"], np.ndarray)
    assert isinstance(info["episode"]["dr"], np.ndarray)
    # Episode records are vectorized because multiple environments
    assert info["episode"]["r"].shape == (num_envs, 2)
    assert info["episode"]["dr"].shape == (num_envs, 2)
    assert isinstance(info["episode"]["l"], np.ndarray)
    assert isinstance(info["episode"]["t"], np.ndarray)
