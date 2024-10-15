import pickle

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils.env_checker import check_env, data_equivalence
from gymnasium.utils.env_match import check_environments_match

import mo_gymnasium as mo_gym


all_testing_env_specs = []
for env_spec in gym.envs.registry.values():
    if type(env_spec.entry_point) is not str:
        continue

    # collect MO Gymnasium envs
    if env_spec.entry_point.split(".")[0] == "mo_gymnasium":
        all_testing_env_specs.append(env_spec)


@pytest.mark.parametrize(
    "spec",
    all_testing_env_specs,
    ids=[spec.id for spec in all_testing_env_specs],
)
def test_all_env_api(spec):
    """Check that all environments pass the environment checker."""
    env = mo_gym.make(spec.id)
    env = mo_gym.wrappers.LinearReward(env)
    check_env(env, skip_render_check=True)
    _test_reward_bounds(env.unwrapped)
    _test_pickle_env(env)


@pytest.mark.parametrize("spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs])
def test_all_env_passive_env_checker(spec):
    env = mo_gym.make(spec.id)
    env.reset()
    env.step(env.action_space.sample())
    env.close()


@pytest.mark.parametrize(
    "gym_id, mo_gym_id",
    [
        ("MountainCar-v0", "mo-mountaincar-v0"),
        ("MountainCarContinuous-v0", "mo-mountaincarcontinuous-v0"),
        ("LunarLander-v3", "mo-lunar-lander-v3"),
        # ("Reacher-v4", "mo-reacher-v4"),  # use a different model and action space
        ("Hopper-v4", "mo-hopper-v4"),
        ("HalfCheetah-v4", "mo-halfcheetah-v4"),
        ("Walker2d-v4", "mo-walker2d-v4"),
        ("Ant-v4", "mo-ant-v4"),
        ("Swimmer-v4", "mo-swimmer-v4"),
        ("Humanoid-v4", "mo-humanoid-v4"),
    ],
)
def test_gymnasium_equivalence(gym_id, mo_gym_id, num_steps=100, seed=123):
    env = gym.make(gym_id)
    mo_env = mo_gym.wrappers.LinearReward(mo_gym.make(mo_gym_id))

    # for float rewards, then precision becomes an issue
    env = gym.wrappers.TransformReward(env, lambda reward: round(reward, 4))
    mo_env = gym.wrappers.TransformReward(mo_env, lambda reward: round(reward, 4))

    check_environments_match(env, mo_env, num_steps=num_steps, seed=seed, skip_rew=True, info_comparison="keys-superset")


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

    env_1 = mo_gym.make(env_spec.id)
    env_2 = mo_gym.make(env_spec.id)
    env_1 = mo_gym.wrappers.LinearReward(env_1)
    env_2 = mo_gym.wrappers.LinearReward(env_2)

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


def _test_reward_bounds(env: gym.Env):
    """Test that the reward bounds are respected."""
    assert env.unwrapped.reward_dim is not None
    assert env.unwrapped.reward_space is not None
    env.reset()
    for _ in range(NUM_STEPS):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        assert env.unwrapped.reward_space.contains(reward)
        if terminated or truncated:
            env.reset()


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
    assert type(a) is type(b), f"{prefix}Differing types: {a} and {b}"
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


def test_ccs_dst():
    env = mo_gym.make("deep-sea-treasure-v0")

    # Known for gamma=0.99
    known_ccs = [
        np.array([0.7, -1.0]),
        np.array([8.037, -2.97]),
        np.array([11.0469, -4.901]),
        np.array([13.181, -6.793]),
        np.array([14.074, -7.726]),
        np.array([14.856, -8.648]),
        np.array([17.3731, -12.2479]),
        np.array([17.814, -13.125]),
        np.array([19.073, -15.706]),
        np.array([19.778, -17.383]),
    ]

    discounted_front = env.unwrapped.pareto_front(gamma=0.99)
    for desired, actual in zip(known_ccs, discounted_front):
        np.testing.assert_array_almost_equal(desired, actual, decimal=2)


def test_ccs_dst_no_discount():
    env = mo_gym.make("deep-sea-treasure-v0")

    known_ccs = mo_gym.envs.deep_sea_treasure.deep_sea_treasure.CONVEX_FRONT

    discounted_front = env.unwrapped.pareto_front(gamma=1.0)
    for desired, actual in zip(known_ccs, discounted_front):
        np.testing.assert_array_almost_equal(desired, actual, decimal=2)


def test_concave_pf_dst():
    env = mo_gym.make("deep-sea-treasure-concave-v0")

    # Known for gamma=0.99
    gamma = 0.99
    known_pf = [
        np.array([1.0, -1.0]),
        np.array([2.0 * gamma**2, -2.97]),
        np.array([3.0 * gamma**4, -4.901]),
        np.array([5.0 * gamma**6, -6.793]),
        np.array([8.0 * gamma**7, -7.726]),
        np.array([16.0 * gamma**8, -8.648]),
        np.array([24.0 * gamma**12, -12.2479]),
        np.array([50.0 * gamma**13, -13.125]),
        np.array([74.0 * gamma**16, -15.706]),
        np.array([124.0 * gamma**18, -17.383]),
    ]

    discounted_front = env.unwrapped.pareto_front(gamma=0.99)
    for desired, actual in zip(known_pf, discounted_front):
        np.testing.assert_array_almost_equal(desired, actual, decimal=2)


def test_concave_pf_dst_no_discount():
    env = mo_gym.make("deep-sea-treasure-concave-v0")

    known_pf = mo_gym.envs.deep_sea_treasure.deep_sea_treasure.CONCAVE_FRONT

    discounted_front = env.unwrapped.pareto_front(gamma=1.0)
    for desired, actual in zip(known_pf, discounted_front):
        np.testing.assert_array_almost_equal(desired, actual, decimal=2)


def test_pf_fruit_tree():
    env = mo_gym.make("fruit-tree-v0")
    depth = 6

    known_pf = np.array(mo_gym.envs.fruit_tree.fruit_tree.FRUITS[str(depth)]) * (0.99 ** (depth - 1))

    discounted_front = env.unwrapped.pareto_front(gamma=0.99)
    for desired, actual in zip(known_pf, discounted_front):
        np.testing.assert_array_almost_equal(desired, actual, decimal=2)


def test_pf_fruit_tree_no_discount():
    env = mo_gym.make("fruit-tree-v0")
    depth = 6

    known_pf = mo_gym.envs.fruit_tree.fruit_tree.FRUITS[str(depth)]

    discounted_front = env.unwrapped.pareto_front(gamma=1.0)
    for desired, actual in zip(known_pf, discounted_front):
        np.testing.assert_array_almost_equal(desired, actual, decimal=2)
