from gymnasium.envs.registration import register


register(
    id="four-room-v0",
    entry_point="mo_gymnasium.envs.four_room.four_room:FourRoom",
    max_episode_steps=200,
)
