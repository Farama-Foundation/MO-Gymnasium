import gym
import numpy as np

import mo_gym
from gym.utils.env_checker import check_env
from mo_gym import MONormalizeReward


def test_normalization_wrapper():
    def finish_episode(env):
        env.reset()
        # Goes to (8.2, -3)
        env.step(3)  # right
        env.step(1)  # down
        _, rewards, _, _ = env.step(1)
        return rewards

    env = mo_gym.make('deep-sea-treasure-v0')
    # Normalize both rewards
    norm_treasure_env = MONormalizeReward(env, idx=0)
    both_norm_env = MONormalizeReward(norm_treasure_env, idx=1)
    for i in range(30):
        finish_episode(both_norm_env)
    both_norm_env.reset()
    _, rewards, _, _ = both_norm_env.step(1)
    np.testing.assert_allclose(rewards, [0.18, -1.24], rtol=0, atol=1e-2)
    rewards = finish_episode(both_norm_env)
    np.testing.assert_allclose(rewards, [2.13, -1.24], rtol=0, atol=1e-2)

    # Normalize only treasure
    for i in range(30):
        finish_episode(norm_treasure_env)
    norm_treasure_env.reset()
    _, rewards, _, _ = norm_treasure_env.step(1)
    # Steps are not normalized
    np.testing.assert_allclose(rewards, [0.18, -1.], rtol=0, atol=1e-2)
    rewards = finish_episode(norm_treasure_env)
    np.testing.assert_allclose(rewards, [2.13, -1.], rtol=0, atol=1e-2)
