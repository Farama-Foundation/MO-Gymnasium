from gym.envs.registration import register
from importlib_metadata import entry_points


register(
    id='mo-halfcheetah-v4',
    entry_point='mo_gym.mujoco.half_cheetah:MOHalfCheehtahEnv',
    max_episode_steps=1000,
)

register(
    id='mo-hopper-v4',
    entry_point='mo_gym.mujoco.hopper:MOHopperEnv',
    max_episode_steps=1000,
)