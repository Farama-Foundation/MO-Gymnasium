from gymnasium.envs.registration import register


register(
    id="mo-halfcheetah-v4",
    entry_point="mo_gymnasium.envs.mujoco.half_cheetah_v4:MOHalfCheehtahEnv",
    max_episode_steps=1000,
)

register(
    id="mo-halfcheetah-v5",
    entry_point="mo_gymnasium.envs.mujoco.half_cheetah_v5:MOHalfCheehtahEnv",
    max_episode_steps=1000,
)

register(
    id="mo-hopper-v4",
    entry_point="mo_gymnasium.envs.mujoco.hopper_v4:MOHopperEnv",
    max_episode_steps=1000,
)

register(
    id="mo-hopper-v5",
    entry_point="mo_gymnasium.envs.mujoco.hopper_v5:MOHopperEnv",
    max_episode_steps=1000,
)

register(
    id="mo-hopper-2d-v4",
    entry_point="mo_gymnasium.envs.mujoco.hopper_v4:MOHopperEnv",
    max_episode_steps=1000,
    kwargs={"cost_objective": False},
)

register(
    id="mo-hopper-2d-v5",
    entry_point="mo_gymnasium.envs.mujoco.hopper_v5:MOHopperEnv",
    max_episode_steps=1000,
    kwargs={"cost_objective": False},
)

register(
    id="mo-walker2d-v4",
    entry_point="mo_gymnasium.envs.mujoco.walker2d:MOWalker2dEnv",
    max_episode_steps=1000,
)

register(
    id="mo-walker2d-v5",
    entry_point="mo_gymnasium.envs.mujoco.walker2d_v5:MOWalker2dEnv",
    max_episode_steps=1000,
)

register(
    id="mo-ant-v4",
    entry_point="mo_gymnasium.envs.mujoco.ant:MOAntEnv",
    max_episode_steps=1000,
)

register(
    id="mo-ant-2d-v4",
    entry_point="mo_gymnasium.envs.mujoco.ant:MOAntEnv",
    max_episode_steps=1000,
    kwargs={"cost_objective": False},
)


register(
    id="mo-ant-v5",
    entry_point="mo_gymnasium.envs.mujoco.ant_v5:MOAntEnv",
    max_episode_steps=1000,
)

register(
    id="mo-ant-2d-v5",
    entry_point="mo_gymnasium.envs.mujoco.ant_v5:MOAntEnv",
    max_episode_steps=1000,
    kwargs={"cost_objective": False},
)

register(
    id="mo-swimmer-v4",
    entry_point="mo_gymnasium.envs.mujoco.swimmer:MOSwimmerEnv",
    max_episode_steps=1000,
)

register(
    id="mo-swimmer-v5",
    entry_point="mo_gymnasium.envs.mujoco.swimmer_v5:MOSwimmerEnv",
    max_episode_steps=1000,
)

register(
    id="mo-humanoid-v4",
    entry_point="mo_gymnasium.envs.mujoco.humanoid:MOHumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="mo-humanoid-v5",
    entry_point="mo_gymnasium.envs.mujoco.humanoid_v5:MOHumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="mo-reacher-v4",
    entry_point="mo_gymnasium.envs.mujoco.reacher_v4:MOReacherEnv",
    max_episode_steps=50,
)

register(
    id="mo-reacher-v5",
    entry_point="mo_gymnasium.envs.mujoco.reacher_v5:MOReacherEnv",
    max_episode_steps=50,
)
