from gymnasium.envs.registration import register


register(
    id="fishwood-v0",
    entry_point="mo_gymnasium.envs.fishwood.fishwood:FishWood",
)
