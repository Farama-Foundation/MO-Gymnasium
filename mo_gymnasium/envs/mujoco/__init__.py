from gymnasium.envs.registration import register


register(
    id="mo-halfcheetah-v4",
    entry_point="mo_gymnasium.envs.mujoco.half_cheetah:MOHalfCheehtahEnv",
    max_episode_steps=1000,
)

register(
    id="mo-hopper-v4",
    entry_point="mo_gymnasium.envs.mujoco.hopper:MOHopperEnv",
    max_episode_steps=1000,
)

register(
    id="mo-hopper-2d-v4",
    entry_point="mo_gymnasium.envs.mujoco.hopper:MOHopperEnv",
    max_episode_steps=1000,
    kwargs={"cost_objective": False},
)

register(
    id="mo-reacher-v4",
    entry_point="mo_gymnasium.envs.mujoco.reacher:MOReacherEnv",
    max_episode_steps=50,
)
