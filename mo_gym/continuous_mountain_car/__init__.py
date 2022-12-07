from gym.envs.registration import register

register(
    id="mo-MountainCarContinuous-v0",
    entry_point="mo_gym.continuous_mountain_car.continuous_mountain_car:MOContinuousMountainCar",
    max_episode_steps=999,
)
