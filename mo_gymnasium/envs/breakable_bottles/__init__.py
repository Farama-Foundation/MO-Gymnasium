from gymnasium.envs.registration import register


register(
    id="breakable-bottles-v0",
    entry_point="mo_gymnasium.envs.breakable_bottles.breakable_bottles:BreakableBottles",
    max_episode_steps=100,
)
