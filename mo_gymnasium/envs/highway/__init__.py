from gymnasium.envs.registration import register


register(id="mo-highway-v0", entry_point="mo_gymnasium.envs.highway.highway:MOHighwayEnv", nondeterministic=True)

register(id="mo-highway-fast-v0", entry_point="mo_gymnasium.envs.highway.highway:MOHighwayEnvFast", nondeterministic=True)

register(id="mo-intersection-v0", entry_point="mo_gymnasium.envs.highway.intersection:MOIntersectionEnv", nondeterministic=True)

register(id="mo-merge-v0", entry_point="mo_gymnasium.envs.highway.merge:MOMergeEnv", nondeterministic=True)

register(id="mo-racetrack-v0", entry_point="mo_gymnasium.envs.highway.racetrack:MORacetrackEnv", nondeterministic=True)

register(id="mo-roundabout-v0", entry_point="mo_gymnasium.envs.highway.roundabout:MORoundaboutEnv", nondeterministic=True)

