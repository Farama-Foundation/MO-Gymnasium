from gym.envs.registration import register


register(
    id='mo-supermario-v0',
    entry_point='mo_gym.mario.mario:MOSuperMarioBros',
    nondeterministic=True,
)