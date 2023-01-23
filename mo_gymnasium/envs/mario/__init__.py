from gymnasium.envs.registration import register


register(
    id="mo-supermario-v0",
    entry_point="mo_gymnasium.envs.mario.mario:MOSuperMarioBros",
    nondeterministic=True,
)
