from gymnasium.envs.registration import register


register(
    id="deep-sea-treasure-v0",
    entry_point="mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
)
