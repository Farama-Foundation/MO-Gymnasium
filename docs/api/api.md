---
title: "API"
---

# API
The environments follow the standard [gymnasium's API](https://github.com/Farama-Foundation/Gymnasium), but return vectorized rewards as numpy arrays.

Here is a minimal example of how to create an environment and interact with it.
```python
import gym
import mo_gym

env = mo_gym.make('minecart-v0') # It follows the original gym's API ...

obs = env.reset()
next_obs, vector_reward, terminated, truncated, info = env.step(your_agent.act(obs))  # but vector_reward is a numpy array!

# Optionally, you can scalarize the reward function with the LinearReward wrapper
env = mo_gym.LinearReward(env, weight=np.array([0.8, 0.2, 0.2]))
```

[![MO-Gym Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucasAlegre/mo-gym/blob/main/mo_gym_demo.ipynb)
You can also check more examples in this colab notebook!
