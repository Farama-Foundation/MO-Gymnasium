from gymnasium.envs.registration import register


register(
    id="mo-supermario-v1",
    entry_point="mo_gymnasium.envs.mario.mario:MOSuperMarioBros",
    max_episode_steps=5000,
    nondeterministic=True,
)
