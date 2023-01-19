from gymnasium.envs.registration import register


register(
    id="resource-gathering-v0",
    entry_point="mo_gymnasium.envs.resource_gathering.resource_gathering:ResourceGathering",
    max_episode_steps=100,
)
