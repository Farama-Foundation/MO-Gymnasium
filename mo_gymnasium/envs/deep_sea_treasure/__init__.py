from gymnasium.envs.registration import register

from mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure import (
    CONCAVE_MAP,
    MIRRORED_MAP,
)


register(
    id="deep-sea-treasure-v0",
    entry_point="mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
)

register(
    id="deep-sea-treasure-concave-v0",
    entry_point="mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": CONCAVE_MAP},
)

register(
    id="deep-sea-treasure-mirrored-v0",
    entry_point="mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure:DeepSeaTreasure",
    max_episode_steps=100,
    kwargs={"dst_map": MIRRORED_MAP},
)
