from gymnasium.envs.registration import register


register(
    id="mo-mountaincar-timemove-v0",
    entry_point="mo_gymnasium.envs.mountain_car_timemove.mountain_car_timemove:MOMountainCar",
    max_episode_steps=200,
)
