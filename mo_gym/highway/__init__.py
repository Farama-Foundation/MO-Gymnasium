from gym.envs.registration import register

register(id="mo-highway-v0", entry_point="mo_gym.highway.highway:MOHighwayEnv")

register(id="mo-highway-fast-v0", entry_point="mo_gym.highway.highway:MOHighwayEnvFast")
