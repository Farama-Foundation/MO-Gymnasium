import numpy as np
import pytest

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
    env.step(3)  # action: right, rewards: [0, -1]
    env.step(1)  # action: down, rewards: [0, -1]
    _, rewards, _, _, infos = env.step(1)  # action: down, rewards: [8.2, -1]
    return rewards, infos


def test_normalization_wrapper():
    # Watch out that the wrapper does not normalize the rewards to have a mean of 0 and std of 1
    # instead it smoothens the moving average of the rewards
    env = mo_gym.make("deep-sea-treasure-v0")
    norm_treasure_env = MONormalizeReward(env, idx=0)
    both_norm_env = MONormalizeReward(norm_treasure_env, idx=1)

    # No normalization
    env.reset(seed=0)
    _, rewards, _, _, _ = env.step(1)
    np.testing.assert_almost_equal(rewards, [0.7, -1.0], decimal=2)

    # Tests for both rewards normalized
    for i in range(30):
        go_to_8_3(both_norm_env)
    both_norm_env.reset(seed=0)
    _, rewards, _, _, _ = both_norm_env.step(1)  # down
    np.testing.assert_almost_equal(rewards, [0.5, -1.24], decimal=2)
    rewards, _ = go_to_8_3(both_norm_env)
    np.testing.assert_almost_equal(rewards, [4.73, -1.24], decimal=2)

    # Tests for only treasure normalized
    for i in range(30):
        go_to_8_3(norm_treasure_env)
    norm_treasure_env.reset(seed=0)
    _, rewards, _, _, _ = norm_treasure_env.step(1)  # down
    # Time rewards are not normalized (-1)
    np.testing.assert_almost_equal(rewards, [0.51, -1.0], decimal=2)
    rewards, _ = go_to_8_3(norm_treasure_env)
    np.testing.assert_almost_equal(rewards, [5.33, -1.0], decimal=2)


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


@pytest.mark.parametrize("idx", [2, 7, -3, -9])
def test_reward_index_outside_the_reward_vector(idx):
    """An index that cannot address a reward component is rejected on construction.

    `deep-sea-treasure-v0` has two of them, so anything outside [-2, 1] used to be
    accepted and then raised `IndexError` from inside numpy several steps later.
    """
    with pytest.raises(ValueError, match="reward components"):
        MONormalizeReward(mo_gym.make("deep-sea-treasure-v0"), idx=idx)

    with pytest.raises(ValueError, match="reward components"):
        MOClipReward(mo_gym.make("deep-sea-treasure-v0"), idx=idx, min_r=0, max_r=1)


@pytest.mark.parametrize("idx", [0, 1, -1, -2])
def test_reward_index_inside_the_reward_vector(idx):
    """Every valid index keeps working, negative ones included."""
    env = MONormalizeReward(mo_gym.make("deep-sea-treasure-v0"), idx=idx)
    env.reset(seed=0)
    env.step(1)
    env.close()


@pytest.mark.parametrize("gamma", [-1.0, -0.01, 1.01, 2.0, 99])
def test_normalize_reward_rejects_gamma_outside_unit_interval(gamma):
    """The accumulator is multiplied by `gamma` every step, so it diverges outside [0, 1]."""
    with pytest.raises(ValueError, match="interval"):
        MONormalizeReward(mo_gym.make("deep-sea-treasure-v0"), idx=0, gamma=gamma)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.99, 1.0])
def test_normalize_reward_accepts_gamma_in_unit_interval(gamma):
    """Both endpoints are meaningful: 0 keeps only the immediate reward, 1 is undiscounted."""
    env = MONormalizeReward(mo_gym.make("deep-sea-treasure-v0"), idx=0, gamma=gamma)
    assert env.gamma == gamma
    env.close()


@pytest.mark.parametrize("epsilon", [0.0, -1e-8, -1.0])
def test_normalize_reward_rejects_non_positive_epsilon(epsilon):
    """`epsilon` sits under a square root to keep the division away from zero."""
    with pytest.raises(ValueError, match="strictly positive"):
        MONormalizeReward(mo_gym.make("deep-sea-treasure-v0"), idx=0, epsilon=epsilon)


def test_clip_reward_rejects_crossed_bounds():
    """`np.clip` with crossed bounds pins the component to `max_r` instead of clipping."""
    with pytest.raises(ValueError, match="greater than"):
        MOClipReward(mo_gym.make("deep-sea-treasure-v0"), idx=0, min_r=5.0, max_r=1.0)

    # Equal bounds are a legitimate way to pin a component and stay accepted.
    env = MOClipReward(mo_gym.make("deep-sea-treasure-v0"), idx=0, min_r=1.0, max_r=1.0)
    env.close()


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
    assert isinstance(info["episode"]["l"], int)
    assert info["episode"]["l"] == 3
    assert isinstance(info["episode"]["t"], float)
