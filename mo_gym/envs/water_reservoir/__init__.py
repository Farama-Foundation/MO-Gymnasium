from gym.envs.registration import register


register(id="water-reservoir-v0", entry_point="mo_gym.envs.water_reservoir.dam_env:DamEnv")
