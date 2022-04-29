from gym.envs.registration import register


register(
    id='mo-reacher-v0',
    entry_point='mo_gym.reacher.reacher:ReacherBulletEnv',
    max_episode_steps=100
)