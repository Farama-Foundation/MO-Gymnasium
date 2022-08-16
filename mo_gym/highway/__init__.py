from gym.envs.registration import register


register(
    id='mo-highway-v0',
    entry_point='mo_gym.highway.highway:MOHighwayEnv',
    max_episode_steps=40
)