from gym.envs.registration import register


register(
    id='fruit-tree-v0',
    entry_point='mo_gym.fruit_tree.fruit_tree:FruitTreeEnv'
)