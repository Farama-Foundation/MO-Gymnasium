---
title: Creating a custom environment
firstpage:
---

## Creating a custom environment

This tutorials goes through the steps of creating a custom environment for MO-Gymnasium. Since MO-Gymnasium is closely tied to Gymnasium, we will refer to its documentation for some parts.

### 1. Create a new environment class
Create an environment class that inherits from [`gymnasium.Env`](https://gymnasium.farama.org/api/env/). The class must implement the following methods:
* `__init__(self, ...)` - The constructor of the class. It should initialize the environment and set the `self.action_space` and`self.observation_space` attributes as in classical Gymnasium (see [Spaces](https://gymnasium.farama.org/api/spaces/). Moreover, since we are dealing with multiple objective/rewards, you should define a `self.reward_space` attribute that defines the shape of the vector rewards returned by the environment, as well as `self.reward_dim` which is an integer defining the size of the reward vector.
* `reset(self, seed, **kwargs)` - Resets the environment and returns the initial observation and info.
* `step(self, action)` - Performs a step in the environment and returns the next observation, vector reward, termination flag, truncated flag, and info. The vector reward should be a numpy array of shape `self.reward_space.shape`.
* `render(self)` - Renders the environment.
* (optional) `pareto_front(self, gamma: float)` - Returns the discounted Pareto front of the environment if known. This is very useful for computing some multi-ojective metrics such as Inverted Generational Distance (IGD).

### 2. Register the environment
Register the environment in the [registry](https://gymnasium.farama.org/api/registry/). This is done by adding the following line to the `__init__.py` file in your env directory:
```python
from gymnasium.envs.registration import register
register(
    id='my_env_v0',
    entry_point='mo_gymnasium.envs.my_env_dir.my_env_file:MyEnv',
)
```

### 3. Test the environment
If your environment is registered within the MO-Gymnasium repository (step 2), it should be automatically pulled for testing when you run `pytest`.

### 4. Instantiate your environment
See our [API documentation](https://mo-gymnasium.farama.org/introduction/api/), but essentially:
```python
import mo_gymnasium
env = mo_gymnasium.make('my_env_v0')
```
