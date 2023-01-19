from gymnasium.envs.registration import register


register(
    id="mo-mountaincarcontinuous-v0",
    entry_point="mo_gymnasium.envs.continuous_mountain_car.continuous_mountain_car:MOContinuousMountainCar",
    max_episode_steps=999,
)
