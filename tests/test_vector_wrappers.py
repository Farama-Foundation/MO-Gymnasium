import numpy as np

import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector import (
    MORecordEpisodeStatistics,
    MOSyncVectorEnv,
)


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
