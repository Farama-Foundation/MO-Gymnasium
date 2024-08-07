import gymnasium as gym
import numpy as np

import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector import MORecordEpisodeStatistics, MOSyncVectorEnv


def test_mo_sync_wrapper():
    num_envs = 3
    envs = MOSyncVectorEnv([lambda: mo_gym.make("deep-sea-treasure-v0") for _ in range(num_envs)])

    envs.reset()
    obs, rewards, terminateds, truncateds, infos = envs.step(envs.action_space.sample())
    assert len(obs) == num_envs, "Number of observations do not match the number of envs"
    assert len(rewards) == num_envs, "Number of rewards do not match the number of envs"
    assert len(terminateds) == num_envs, "Number of terminateds do not match the number of envs"
    assert len(truncateds) == num_envs, "Number of truncateds do not match the number of envs"
    envs.close()


def test_mo_sync_autoreset():
    num_envs = 2
    envs = MOSyncVectorEnv([lambda: mo_gym.make("deep-sea-treasure-v0") for _ in range(num_envs)])

    obs, infos = envs.reset()
    assert (obs[0] == [0, 0]).all()
    assert (obs[1] == [0, 0]).all()
    obs, rewards, terminateds, truncateds, infos = envs.step([0, 1])
    assert (obs[0] == [0, 0]).all()
    assert (obs[1] == [1, 0]).all()
    # Use np assert almost equal to avoid floating point errors
    np.testing.assert_almost_equal(rewards[0], np.array([0.0, -1.0], dtype=np.float32), decimal=2)
    np.testing.assert_almost_equal(rewards[1], np.array([0.7, -1.0], dtype=np.float32), decimal=2)
    assert not terminateds[0]
    assert terminateds[1]  # This one is done
    assert not truncateds[0]
    assert not truncateds[1]
    obs, rewards, terminateds, truncateds, infos = envs.step([0, 1])
    assert (obs[0] == [0, 0]).all()
    assert (obs[1] == [0, 0]).all()
    assert (rewards[0] == [0.0, -1.0]).all()
    assert (rewards[1] == [0.0, 0.0]).all()  # Reset step
    assert not terminateds[0]
    assert not terminateds[1]  # Not done anymore
    envs.close()


def test_mo_record_ep_statistic_vector_env():
    num_envs = 2
    envs = MOSyncVectorEnv([lambda: mo_gym.make("deep-sea-treasure-v0") for _ in range(num_envs)])
    envs = MORecordEpisodeStatistics(envs, gamma=0.97)

    envs.reset()
    terminateds = np.array([False] * num_envs)
    info = {}
    obs, rewards, terminateds, _, info = envs.step([0, 3])
    obs, rewards, terminateds, _, info = envs.step([0, 1])
    obs, rewards, terminateds, _, info = envs.step([0, 1])

    assert isinstance(info["episode"]["r"], np.ndarray)
    assert isinstance(info["episode"]["dr"], np.ndarray)
    # Episode records are vectorized because multiple environments
    assert info["episode"]["r"].shape == (num_envs, 2)
    np.testing.assert_almost_equal(info["episode"]["r"][0], np.array([0.0, 0.0], dtype=np.float32), decimal=2)
    np.testing.assert_almost_equal(info["episode"]["r"][1], np.array([8.2, -3.0], dtype=np.float32), decimal=2)
    assert info["episode"]["dr"].shape == (num_envs, 2)
    np.testing.assert_almost_equal(info["episode"]["dr"][0], np.array([0.0, 0.0], dtype=np.float32), decimal=2)
    np.testing.assert_almost_equal(info["episode"]["dr"][1], np.array([7.72, -2.91], dtype=np.float32), decimal=2)
    assert isinstance(info["episode"]["l"], np.ndarray)
    np.testing.assert_almost_equal(info["episode"]["l"], np.array([0, 3], dtype=np.float32), decimal=2)
    assert isinstance(info["episode"]["t"], np.ndarray)
    envs.close()


def test_gym_wrapper_and_vector():
    # This tests the integration of gym-wrapped envs with MO-Gymnasium vectorized envs
    num_envs = 2
    envs = MOSyncVectorEnv(
        [lambda: gym.wrappers.NormalizeObservation(mo_gym.make("deep-sea-treasure-v0")) for _ in range(num_envs)]
    )

    envs.reset()
    for i in range(30):
        obs, rewards, terminateds, truncateds, infos = envs.step(envs.action_space.sample())
    assert len(obs) == num_envs, "Number of observations do not match the number of envs"
    assert len(rewards) == num_envs, "Number of rewards do not match the number of envs"
    assert len(terminateds) == num_envs, "Number of terminateds do not match the number of envs"
    assert len(truncateds) == num_envs, "Number of truncateds do not match the number of envs"
    envs.close()
