from gymnasium.envs.registration import register


register(
    id="mo-lunar-lander-v3",
    entry_point="mo_gymnasium.envs.lunar_lander.lunar_lander:MOLunarLander",
    max_episode_steps=1000,
)

register(
    id="mo-lunar-lander-continuous-v3",
    entry_point="mo_gymnasium.envs.lunar_lander.lunar_lander:MOLunarLander",
    max_episode_steps=1000,
    kwargs={"continuous": True},
)
