![tests](https://github.com/Farama-Foundation/mo-gymnasium/workflows/Python%20tests/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# MO-Gymnasium: Multi-Objective Reinforcement Learning Environments

<!-- start elevator-pitch -->

MO-Gymnasium is an open source Python library for developing and comparing multi-objective reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API. Essentially, the environments follow the standard [Gymnasium API](https://github.com/Farama-Foundation/Gymnasium), but return vectorized rewards as numpy arrays.

The documentation website is at [mo-gymnasium.farama.org](https://mo-gymnasium.farama.org), and we have a public discord server (which we also use to coordinate development work) that you can join here: https://discord.gg/bnJ6kubTg6.

<!-- end elevator-pitch -->

## Environments

<!-- start environments -->
MO-Gymnasium includes environments taken from the MORL literature, as well as multi-objective version of classical environments, such as Mujoco.

| Env                                                                                                                                                                                                                                        | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                                                                                                         |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deep-sea-treasure-v0`](https://mo-gymnasium.farama.org/environments/deep-sea-treasure/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/dst.png" width="200px">                          | Discrete / Discrete                 | `[treasure, time_penalty]`                                    | Agent is a submarine that must collect a treasure while taking into account a time penalty. Treasures values taken from [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf).                                                                                                                                                                   |
| [`resource-gathering-v0`](https://mo-gymnasium.farama.org/environments/resource-gathering/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/resource-gathering.png" width="200px">         | Discrete / Discrete                 | `[enemy, gold, gem]`                                          | Agent must collect gold or gem. Enemies have a 10% chance of killing the agent. From [Barret & Narayanan 2008](https://dl.acm.org/doi/10.1145/1390156.1390162).                                                                                                                                                                                     |
| [`fishwood-v0`](https://mo-gymnasium.farama.org/environments/fishwood/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/fishwood.png" width="200px">                                       | Discrete / Discrete                 | `[fish_amount, wood_amount]`                                  | ESR environment, the agent must collect fish and wood to light a fire and eat. From [Roijers et al. 2018](https://www.researchgate.net/publication/328718263_Multi-objective_Reinforcement_Learning_for_the_Expected_Utility_of_the_Return).                                                                                                        |
| [`breakable-bottles-v0`](https://mo-gymnasium.farama.org/environments/breakable-bottles/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/breakable-bottles.jpg" width="200px">            | Discrete (Dictionary) / Discrete    | `[time_penalty, bottles_delivered, potential]`                | Gridworld with 5 cells. The agents must collect bottles from the source location and deliver to the destination. From [Vamplew et al. 2021](https://www.sciencedirect.com/science/article/pii/S0952197621000336).                                                                                                                                   |
| [`four-room-v0`](https://mo-gymnasium.farama.org/environments/four-room/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/four-room.png" width="200px">                                    | Discrete / Discrete                 | `[item1, item2, item3]`                                       | Agent must collect three different types of items in the map and reach the goal. From [Alegre et al. 2022](https://proceedings.mlr.press/v162/alegre22a.html).                                                                                                                                                                                      |
| [`mo-mountaincar-v0`](https://mo-gymnasium.farama.org/environments/mo-mountaincar/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/mo-mountaincar.png" width="200px">                     | Continuous / Discrete               | `[time_penalty, reverse_penalty, forward_penalty]`            | Classic Mountain Car env, but with extra penalties for the forward and reverse actions. From [Vamplew et al. 2011](https://www.researchgate.net/publication/220343783_Empirical_evaluation_methods_for_multiobjective_reinforcement_learning_algorithms).                                                                                           |
| [`mo-mountaincarcontinuous-v0`](https://mo-gymnasium.farama.org/environments/mo-mountaincarcontinuous/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/mo-mountaincar.png" width="200px"> | Continuous / Continuous             | `[time_penalty, fuel_consumption_penalty]`                    | Continuous Mountain Car env, but with penalties for fuel consumption.                                                                                                                                                                                                                                                                               |
| [`mo-lunar-lander-v2`](https://mo-gymnasium.farama.org/environments/mo-lunar-lander/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/lunarlander.png" width="200px">                      | Continuous / Discrete or Continuous | `[landed, shaped_reward, main_engine_fuel, side_engine_fuel]` | MO version of the `LunarLander-v2` [environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/). Objectives defined similarly as in [Hung et al. 2022](https://openreview.net/forum?id=AwWaBXLIJE).                                                                                                                                 |
| [`minecart-v0`](https://mo-gymnasium.farama.org/environments/minecart/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/minecart.png" width="200px">                                       | Continuous or Image / Discrete      | `[ore1, ore2, fuel]`                                          | Agent must collect two types of ores and minimize fuel consumption. From [Abels et al. 2019](https://arxiv.org/abs/1809.07803v2).                                                                                                                                                                                                                   |
| [`mo-highway-v0`](https://mo-gymnasium.farama.org/environments/mo-highway/) and `mo-highway-fast-v0` <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/highway.png" width="200px">           | Continuous / Discrete               | `[speed, right_lane, collision]`                              | The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles and staying on the rightest lane. From [highway-env](https://github.com/eleurent/highway-env).                                                                                                                                                  |
| [`mo-reacher-v4`](https://mo-gymnasium.farama.org/environments/mo-reacher/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/reacher-mujoco.png" width="200px">                             | Continuous / Discrete               | `[target_1, target_2, target_3, target_4]`                    | Mujoco version of `mo-reacher-v0`, based on `Reacher-v4` [environment](https://gymnasium.farama.org/environments/mujoco/reacher/).                                                                                                                                                                                                                  |
| [`mo-halfcheetah-v4`](https://mo-gymnasium.farama.org/environments/mo-halfcheetah/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/cheetah.png" width="200px">                            | Continuous / Continuous             | `[velocity, energy]`                                          | Multi-objective version of [HalfCheetah-v4](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) env. Similar to [Xu et al. 2020](https://github.com/mit-gfx/PGMORL).                                                                                                                                                                    |
| [`mo-hopper-v4`](https://mo-gymnasium.farama.org/environments/mo-hopper/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/hopper.png" width="200px">                                       | Continuous / Continuous             | `[velocity, height, energy]`                                  | Multi-objective version of [Hopper-v4](https://gymnasium.farama.org/environments/mujoco/hopper/) env.                                                                                                                                                                                                                                               |
| [`water-reservoir-v0`](https://mo-gymnasium.farama.org/environments/water-reservoir/)                                                                                                                                                      | Continuous / Continuous             | `[cost_flooding, deficit_water]`                              | A Water reservoir environment. The agent executes a continuous action, corresponding to the amount of water released by the dam. From [Pianosi et al. 2013](https://iwaponline.com/jh/article/15/2/258/3425/Tree-based-fitted-Q-iteration-for-multi-objective).                                                                                     |
| [`fruit-tree-v0`](https://mo-gymnasium.farama.org/environments/fruit-tree/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/fruit-tree.png" width="200px">                                 | Discrete / Discrete                 | `[nutri1, ..., nutri6]`                                       | Full binary tree of depth d=5,6 or 7. Every leaf contains a fruit with a value for the nutrients Protein, Carbs, Fats, Vitamins, Minerals and Water. From [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf).                                                                                                                                 |
| [`mo-reacher-v0`]() <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/reacher.png" width="200px">                                                                                            | Continuous / Discrete               | `[target_1, target_2, target_3, target_4]`                    | [:warning: PyBullet support is limited.] Reacher robot from [PyBullet](https://github.com/benelot/pybullet-gym/blob/ec9e87459dd76d92fe3e59ee4417e5a665504f62/pybulletgym/envs/roboschool/robots/manipulators/reacher.py), but there are 4 different target positions. From [Alegre et al. 2022](https://proceedings.mlr.press/v162/alegre22a.html). |
| [`mo-supermario-v0`](https://mo-gymnasium.farama.org/environments/mo-supermario/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/screenshots/mario.png" width="200px">                                | Image / Discrete                    | `[x_pos, time, death, coin, enemy]`                           | [:warning: SuperMarioBrosEnv support is limited.] Multi-objective version of [SuperMarioBrosEnv](https://github.com/Kautenja/gym-super-mario-bros). Objectives are defined similarly as in [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf).                                                                                                |

<!-- end environments -->

## Installation
<!-- start install -->

Via pip:
```bash
pip install mo-gymnasium
```

This does not include dependencies for all families of environments (some can be problematic to install on certain systems). You can install these dependencies for one family like `pip install "mo-gymnasium[mujoco]"` or use `pip install "mo-gymnasium[all]"` to install all dependencies.

Alternatively, you can install the newest unreleased version:
```bash
git clone https://github.com/Farama-Foundation/MO-Gymnasium
cd MO-Gymnasium
pip install -e .
```

<!-- end install -->

## API

<!-- start snippet-usage -->

As for Gymnasium, the MO-Gymnasium API models environments as simple Python `env` classes. Creating environment instances and interacting with them is very simple - here's an example using the "minecart-v0" environment:

```python
import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np

# It follows the original Gymnasium API ...
env = mo_gym.make('minecart-v0')

obs = env.reset()
# but vector_reward is a numpy array!
next_obs, vector_reward, terminated, truncated, info = env.step(your_agent.act(obs))

# Optionally, you can scalarize the reward function with the LinearReward wrapper
env = mo_gym.LinearReward(env, weight=np.array([0.8, 0.2, 0.2]))
```
For details on multi-objective MDP's (MOMDP's) and other MORL definitions, see [A practical guide to multi-objective reinforcement learning and planning](https://link.springer.com/article/10.1007/s10458-022-09552-y).

You can also check more examples in this colab notebook! [![MO-Gym Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Farama-Foundation/MO-Gymnasium/blob/main/mo_gymnasium_demo.ipynb)

<!-- end snippet-usage -->

## Notable related libraries

[MORL-Baselines](https://github.com/LucasAlegre/morl-baselines) is a repository containing various implementations of MORL algorithms by the same authors as MO-Gymnasium. It relies on the MO-Gymnasium API and shows various examples of the usage of wrappers and environments.

## Environment Versioning

MO-Gymnasium keeps strict versioning for reproducibility reasons. All environments end in a suffix like "-v0".  When changes are made to environments that might impact learning results, the number is increased by one to prevent potential confusion.

## Citing

<!-- start citation -->

If you use this repository in your work, please cite:

```bibtex
@inproceedings{Alegre+2022bnaic,
  author = {Lucas N. Alegre and Florian	Felten and El-Ghazali Talbi and Gr{\'e}goire Danoy and Ann Now{\'e} and Ana L. C. Bazzan and Bruno C. da Silva},
  title = {{MO-Gym}: A Library of Multi-Objective Reinforcement Learning Environments},
  booktitle = {Proceedings of the 34th Benelux Conference on Artificial Intelligence BNAIC/Benelearn 2022},
  year = {2022}
}
```

<!-- end citation -->

## Acknowledgments

<!-- start acknowledgments -->

* The `minecart-v0` env is a refactor of https://github.com/axelabels/DynMORL.
* The `deep-sea-treasure-v0`, `fruit-tree-v0` and `mo-supermario-v0` envs are based on https://github.com/RunzheYang/MORL.
* The `four-room-v0` env is based on https://github.com/mike-gimelfarb/deep-successor-features-for-transfer.
* The `fishwood-v0` code was provided by Denis Steckelmacher and Conor F. Hayes.
* The `water-reservoir-v0` code was provided by Mathieu Reymond.

<!-- end acknowledgments -->
