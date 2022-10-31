from gym.envs.registration import register

register(
    id="fishwood-v0",
    entry_point="mo_gym.fishwood.fishwood:FishWood",
)
