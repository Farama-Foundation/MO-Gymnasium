from gym.envs.registration import register


register(
    id='minecart-v0',
    entry_point='mo_gym.minecart.minecart:Minecart',
    max_episode_steps=1000
)