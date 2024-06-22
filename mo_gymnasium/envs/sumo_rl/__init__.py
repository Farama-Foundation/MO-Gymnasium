from gymnasium.envs.registration import register


register(
    id="mo-intersection-v0",
    entry_point="mo_gymnasium.envs.sumo_rl.mo_sumo_rl:MOIntersectionEnv",
    max_episode_steps=300,
)
