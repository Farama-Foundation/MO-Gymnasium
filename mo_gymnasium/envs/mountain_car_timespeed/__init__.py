from gymnasium.envs.registration import register


register(
    id="mo-mountaincar-timespeed-v0",
    entry_point="mo_gymnasium.envs.mountain_car_timespeed.mountain_car_timespeed:MOMountainCar",
    max_episode_steps=200,
)
