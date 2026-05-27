from pathlib import Path

from gymnasium.envs.registration import register


register(
    id="minecart-v1",
    entry_point="mo_gymnasium.envs.minecart.minecart:Minecart",
    max_episode_steps=1000,
)

register(
    id="minecart-rgb-v1",
    entry_point="mo_gymnasium.envs.minecart.minecart:Minecart",
    kwargs={"image_observation": True},
    nondeterministic=True,  # This is a nondeterministic environment due to the random placement of the mines
    max_episode_steps=1000,
)

register(
    id="minecart-deterministic-v1",
    entry_point="mo_gymnasium.envs.minecart.minecart:Minecart",
    kwargs={"config": str(Path(__file__).parent.absolute()) + "/mine_config_det.json"},
    max_episode_steps=1000,
)
