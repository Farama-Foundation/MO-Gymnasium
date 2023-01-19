from gymnasium.envs.registration import register


register(
    id="mo-reacher-v0",
    entry_point="mo_gymnasium.envs.reacher.reacher:ReacherBulletEnv",
    max_episode_steps=100,
    kwargs={"fixed_initial_state": None},
)
