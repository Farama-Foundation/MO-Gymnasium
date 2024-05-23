import numpy as np

import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import (
    MOClipReward,
    MONormalizeReward,
    MORecordEpisodeStatistics,
)


def go_to_8_3(env):
    """
    Goes to (8.2, -3) treasure, returns the rewards
    """
    env.reset()
    env.step(3)  # right
    env.step(1)  # down
    _, rewards, _, _, infos = env.step(1)
    return rewards, infos


def test_normalization_wrapper():
    env = mo_gym.make("deep-sea-treasure-v0")
    norm_treasure_env = MONormalizeReward(env, idx=0)
    both_norm_env = MONormalizeReward(norm_treasure_env, idx=1)

    # No normalization
    env.reset(seed=0)
    _, rewards, _, _, _ = env.step(1)
    np.testing.assert_allclose(rewards, [0.7, -1.0], rtol=0, atol=1e-2)

    # Tests for both rewards normalized
    for i in range(30):
        go_to_8_3(both_norm_env)
    both_norm_env.reset(seed=0)
    _, rewards, _, _, _ = both_norm_env.step(1)  # down
    np.testing.assert_allclose(
        rewards, [0.49, -1.24], rtol=0, atol=1e-2
    )  # TODO PR check why we had to change those values @Mark?
    rewards, _ = go_to_8_3(both_norm_env)
    np.testing.assert_allclose(rewards, [4.73, -1.24], rtol=0, atol=1e-2)

    # Tests for only treasure normalized
    for i in range(30):
        go_to_8_3(norm_treasure_env)
    norm_treasure_env.reset(seed=0)
    _, rewards, _, _, _ = norm_treasure_env.step(1)  # down
    # Time rewards are not normalized (-1)
    np.testing.assert_allclose(rewards, [0.51, -1.0], rtol=0, atol=1e-2)
    rewards, _ = go_to_8_3(norm_treasure_env)
    np.testing.assert_allclose(rewards, [5.33, -1.0], rtol=0, atol=1e-2)


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
    assert tuple(np.round(info["episode"]["dr"], 2)) == (
        np.float32(7.48),
        np.float32(-2.82),
    )
    assert isinstance(info["episode"]["l"], int)
    assert info["episode"]["l"] == 3
    assert isinstance(info["episode"]["t"], float)
