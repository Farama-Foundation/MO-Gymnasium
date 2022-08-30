![tests](https://github.com/LucasAlegre/mo-gym/workflows/Python%20tests/badge.svg)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](https://github.com/LucasAlegre/mo-gym/blob/main/LICENSE)

# MO-Gym: Multi-Objective Reinforcement Learning Environments

Gym environments for multi-objective reinforcement learning (MORL). The environments follow the standard [gym's API](https://github.com/openai/gym), but return vectorized rewards as numpy arrays.

For details on multi-objective MDP's (MOMDP's) and other MORL definitions, see [A practical guide to multi-objective reinforcement learning and planning](https://link.springer.com/article/10.1007/s10458-022-09552-y).

## Install

Via pip:
```bash
pip install mo-gym
```

Alternatively, you can install the newest unreleased version:
```bash
git clone https://github.com/LucasAlegre/mo-gym.git
cd mo-gym
pip install -e .
```

## Usage

```python
import gym
import mo_gym

env = mo_gym.make('minecart-v0') # It follows the original gym's API ...

obs = env.reset()
next_obs, vector_reward, done, info = env.step(your_agent.act(obs))  # but vector_reward is a numpy array!

# Optionally, you can scalarize the reward function with the LinearReward wrapper
env = mo_gym.LinearReward(env, weight=np.array([0.8, 0.2, 0.2]))
```

[![MO-Gym Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LucasAlegre/mo-gym/blob/main/mo_gym_demo.ipynb)
You can also check more examples in this colab notebook! 

## Environments

| Env | Obs/Action spaces | Objectives | Description |
| --- | --- | --- | --- |
|  `deep-sea-treasure-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/dst.png" width="200px"> | Discrete / Discrete |  `[treasure, time_penalty]` | Agent is a submarine that must collect a treasure while taking into account a time penalty. Treasures values taken from [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf). |
|  `resource-gathering-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/resource-gathering.png" width="200px"> | Discrete / Discrete |  `[enemy, gold, gem]` | Agent must collect gold or gem. Enemies have a 10% chance of killing the agent. From [Barret & Narayanan 2008](https://dl.acm.org/doi/10.1145/1390156.1390162). |
|  `fruit-tree-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/fruit-tree.png" width="200px"> | Discrete / Discrete |  `[nutri1, ..., nutri6]` | Full binary tree of depth d=5,6 or 7. Every leaf contains a fruit with a value for the nutrients Protein, Carbs, Fats, Vitamins, Minerals and Water. From [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf). |
|  `breakable-bottles-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/breakable-bottles.jpg" width="200px"> | Discrete (Dictionary) / Discrete |  `[time_penalty, bottles_delivered, potential]` | Gridworld with 5 cells. The agents must collect bottles from the source location and deliver to the destination. From [Vamplew et al. 2021](https://www.sciencedirect.com/science/article/pii/S0952197621000336). |
|  `four-room-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/four-room.png" width="200px"> | Discrete / Discrete |  `[item1, item2, item3]` | Agent must collect three different types of items in the map and reach the goal. From [Alegre et al. 2022](https://proceedings.mlr.press/v162/alegre22a.html). |
|  `mo-mountaincar-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/mo-mountaincar.png" width="200px"> | Continuous / Discrete |  `[time_penalty, reverse_penalty, forward_penalty]` | Classic Mountain Car env, but with extra penalties for the forward and reverse actions. From [Vamplew et al. 2011](https://www.researchgate.net/publication/220343783_Empirical_evaluation_methods_for_multiobjective_reinforcement_learning_algorithms). |
| `mo-reacher-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/reacher.png" width="200px">| Continuous / Discrete | `[target_1, target_2, target_3, target_4]` | Reacher robot from [PyBullet](https://github.com/benelot/pybullet-gym/blob/ec9e87459dd76d92fe3e59ee4417e5a665504f62/pybulletgym/envs/roboschool/robots/manipulators/reacher.py), but there are 4 different target positions. From [Alegre et al. 2022](https://proceedings.mlr.press/v162/alegre22a.html). 
|  `minecart-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/minecart.png" width="200px"> | Continuous or Image / Discrete |  `[ore1, ore2, fuel]`  | Agent must collect two types of ores and minimize fuel consumption. From [Abels et al. 2019](https://arxiv.org/abs/1809.07803v2). |
|  `mo-supermario-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/mario.png" width="200px"> | Image / Discrete |  `[x_pos, time, death, coin, enemy]`  | Multi-objective version of [SuperMarioBrosEnv](https://github.com/Kautenja/gym-super-mario-bros). Objectives are defined similarly as in [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf). |
|  `mo-halfcheetah-v4` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/cheetah.png" width="200px"> | Continuous / Continuous |  `[velocity, energy]`  | Multi-objective version of [HalfCheetah-v4](https://www.gymlibrary.ml/environments/mujoco/half_cheetah/) env. Similar to [Xu et al. 2020](https://github.com/mit-gfx/PGMORL). |
|  `mo-hopper-v4` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/hopper.png" width="200px"> | Continuous / Continuous |  `[velocity, height, energy]`  | Multi-objective version of [Hopper-v4](https://www.gymlibrary.ml/environments/mujoco/hopper/) env. |

## Citing

If you use this repository in your work, please cite:

```bibtex
@misc{mo-gym,
  author = {Lucas N. Alegre and Florian Felten},
  title = {MO-Gym: Multi-Objective Reinforcement Learning Environments},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LucasAlegre/mo-gym}},
}
```

## Acknowledgments

* The `minecart-v0` env is a refactor of https://github.com/axelabels/DynMORL.
* The `deep-sea-treasure-v0`, `fruit-tree-v0` and `mo-supermario-v0` envs are based on https://github.com/RunzheYang/MORL.
* The `four-room-v0` env is based on https://github.com/mike-gimelfarb/deep-successor-features-for-transfer.
