from gym.envs.registration import register


register(
    id="mo-lunar-lander-v2",
    entry_point="mo_gym.envs.lunar_lander.lunar_lander:MOLunarLander",
    max_episode_steps=1000,
)
