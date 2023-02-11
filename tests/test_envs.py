import pickle

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils.env_checker import check_env, data_equivalence

import mo_gymnasium as mo_gym


all_testing_env_specs = []
for env_spec in gym.envs.registry.values():
    # collect MO Gymnasium envs
    if env_spec.entry_point.split(".")[0] == "mo_gymnasium":
        if type(env_spec.entry_point) is not str:
            continue
        # Ignore highway as they do not deal with the random seed appropriately
        if not env_spec.id.startswith("mo-highway"):
            all_testing_env_specs.append(env_spec)


@pytest.mark.parametrize(
    "spec",
    all_testing_env_specs,
    ids=[spec.id for spec in all_testing_env_specs],
)
def test_all_env_api(spec):
    """Check that all environments pass the environment checker."""
    env = mo_gym.make(spec.id)
    env = mo_gym.LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_pickle_env(env)


@pytest.mark.parametrize("spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs])
def test_all_env_passive_env_checker(spec):
    env = mo_gym.make(spec.id)
    env.reset()
    env.step(env.action_space.sample())
    env.close()


# Note that this precludes running this test in multiple threads.
# However, we probably already can't do multithreading due to some environments.
SEED = 0
NUM_STEPS = 50


@pytest.mark.parametrize(
    "env_spec",
    all_testing_env_specs,
    ids=[env.id for env in all_testing_env_specs],
)
def test_env_determinism_rollout(env_spec: EnvSpec):
    """Run a rollout with two environments and assert equality.
    This test run a rollout of NUM_STEPS steps with two environments
    initialized with the same seed and assert that:
    - observation after first reset are the same
    - same actions are sampled by the two envs
    - observations are contained in the observation space
    - obs, rew, done and info are equals between the two envs
    """
    # Don't check rollout equality if it's a nondeterministic environment.
    if env_spec.nondeterministic is True:
        return

    env_1 = env_spec.make(disable_env_checker=True)
    env_2 = env_spec.make(disable_env_checker=True)
    env_1 = mo_gym.LinearReward(env_1)
    env_2 = mo_gym.LinearReward(env_2)

    initial_obs_1, initial_info_1 = env_1.reset(seed=SEED)
    initial_obs_2, initial_info_2 = env_2.reset(seed=SEED)
    assert_equals(initial_obs_1, initial_obs_2)

    env_1.action_space.seed(SEED)

    for time_step in range(NUM_STEPS):
        # We don't evaluate the determinism of actions
        action = env_1.action_space.sample()

        obs_1, rew_1, terminated_1, truncated_1, info_1 = env_1.step(action)
        obs_2, rew_2, terminated_2, truncated_2, info_2 = env_2.step(action)

        assert_equals(obs_1, obs_2, f"[{time_step}] ")
        assert env_1.observation_space.contains(obs_1)  # obs_2 verified by previous assertion

        assert rew_1 == rew_2, f"[{time_step}] reward 1={rew_1}, reward 2={rew_2}"
        assert terminated_1 == terminated_2, f"[{time_step}] done 1={terminated_1}, done 2={terminated_2}"
        assert truncated_1 == truncated_2, f"[{time_step}] done 1={truncated_1}, done 2={truncated_2}"
        assert_equals(info_1, info_2, f"[{time_step}] ")

        if terminated_1 or truncated_1:  # terminated_2, truncated_2 verified by previous assertion
            env_1.reset(seed=SEED)
            env_2.reset(seed=SEED)

    env_1.close()
    env_2.close()


def _test_pickle_env(env: gym.Env):
    pickled_env = pickle.loads(pickle.dumps(env))

    data_equivalence(env.reset(), pickled_env.reset())

    action = env.action_space.sample()
    data_equivalence(env.step(action), pickled_env.step(action))
    env.close()
    pickled_env.close()


def assert_equals(a, b, prefix=None):
    """Assert equality of data structures `a` and `b`.
    Args:
        a: first data structure
        b: second data structure
        prefix: prefix for failed assertion message for types and dicts
    """
    assert type(a) == type(b), f"{prefix}Differing types: {a} and {b}"
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"{prefix}Key sets differ: {a} and {b}"

        for k in a.keys():
            v_a = a[k]
            v_b = b[k]
            assert_equals(v_a, v_b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, tuple):
        for elem_from_a, elem_from_b in zip(a, b):
            assert_equals(elem_from_a, elem_from_b)
    else:
        assert a == b
