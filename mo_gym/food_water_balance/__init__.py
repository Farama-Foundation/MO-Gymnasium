from gym.envs.registration import register


register(
    id='food-water-balance-v0',
    entry_point='mo_gym.food_water_balance.food_water_balance:FoodWaterBalance',
    max_episode_steps=100000
)