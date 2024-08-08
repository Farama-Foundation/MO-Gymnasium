from gymnasium.envs.registration import register


register(
    id="mo-mountaincar-v0",
    entry_point="mo_gymnasium.envs.mountain_car.mountain_car:MOMountainCar",
    max_episode_steps=200,
)

register(
    id="mo-mountaincar-3d-v0",
    entry_point="mo_gymnasium.envs.mountain_car.mountain_car:MOMountainCar",
    max_episode_steps=200,
    kwargs={"add_speed_objective": True, "merge_move_penalty": True},
)

register(
    id="mo-mountaincar-timemove-v0",
    entry_point="mo_gymnasium.envs.mountain_car.mountain_car:MOMountainCar",
    max_episode_steps=200,
    kwargs={"merge_move_penalty": True},
)

register(
    id="mo-mountaincar-timespeed-v0",
    entry_point="mo_gymnasium.envs.mountain_car.mountain_car:MOMountainCar",
    max_episode_steps=200,
    kwargs={"remove_move_penalty": True, "add_speed_objective": True},
)
