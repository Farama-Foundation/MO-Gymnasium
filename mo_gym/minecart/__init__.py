from pathlib import Path

from gym.envs.registration import register

register(
    id='minecart-v0',
    entry_point='mo_gym.minecart.minecart:Minecart',
    max_episode_steps=1000
)

register(
    id='minecart-deterministic-v0',
    entry_point='mo_gym.minecart.minecart:Minecart',
    kwargs={'config': str(Path(__file__).parent.absolute()) + '/mine_config_det.json'},
    max_episode_steps=1000
)
