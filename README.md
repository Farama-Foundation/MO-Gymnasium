![tests](https://github.com/LucasAlegre/mo-gym/actions/workflows/test.yml/badge.svg)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](https://github.com/LucasAlegre/mo-gym/blob/main/LICENSE)

# MO-Gym: Multi-Objective Reinforcement Learning Environments

Gym environments for multi-objective reinforcement learning (MORL). The environments follow the standard [gym's API](https://github.com/openai/gym), but return vectorized rewards as numpy arrays.

For details on multi-objective MPDS (MOMDP's) and other MORL definitions, see [A practical guide to multi-objective reinforcement learning and planning](https://link.springer.com/article/10.1007/s10458-022-09552-y).

## Install
```bash
git clone https://github.com/LucasAlegre/mo-gym.git
cd mo-gym
pip install -e .
```

## Usage

```python
import gym
import mo_gym

env = gym.make('minecart-v0') # It follows the original gym's API ...

obs = env.reset()
next_obs, vector_reward, done, info = env.step(your_agent.act(obs))  # but vector_reward is a numpy array!

# Optionally, you can scalarize the reward function with the LinearReward wrapper
env = mo_gym.LinearReward(env, weight=np.array([0.8, 0.2, 0.2]))
```

## Environments

| Env | Obs/Action spaces | Objectives | Description |
| --- | --- | --- | --- |
|  `deep-sea-treasure-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/dst.png" width="200px"> | Discrete / Discrete |  `[treasure, time_penalty]` | Agent is a submarine that must collect a treasure while taking into account a time penalty. Treasures values taken from [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf). |
|  `resource-gathering-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/resource-gathering.png" width="200px"> | Discrete / Discrete |  `[enemy, gold, gem]` | Agent must collect gold or gem. Enemies have a 10% chance of killing the agent. From [Barret & Narayanan 2008](https://dl.acm.org/doi/10.1145/1390156.1390162). |
|  `four-room-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/four-room.png" width="200px"> | Discrete / Discrete |  `[item1, item2, item3]` | Agent must collect three different types of items in the map and reach the goal.|
|  `mo-mountaincar-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/mo-mountaincar.png" width="200px"> | Continuous / Discrete |  `[time_penalty, reverse_penalty, forward_penalty]` | Classic Mountain Car env, but with extra penalties for the forward and reverse actions. From [Vamplew et al. 2011](https://www.researchgate.net/publication/220343783_Empirical_evaluation_methods_for_multiobjective_reinforcement_learning_algorithms). |
|  `minecart-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/minecart.png" width="200px"> | Continuous or Image / Discrete |  `[ore1, ore2, fuel]`  | Agent must collect two types of ores and minimize fuel consumption. From [Abels et al. 2019](https://arxiv.org/abs/1809.07803v2). |
|  `mo-supermario-v0` <br><img src="https://raw.githubusercontent.com/LucasAlegre/mo-gym/main/screenshots/mario.png" width="200px"> | Image / Discrete |  `[x_pos, time, death, coin, enemy]`  | Multi-objective version of [SuperMarioBrosEnv](https://github.com/Kautenja/gym-super-mario-bros). Objectives are defined similarly as in [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf). |


## Citing

If you use this repository in your work, please cite:

```bibtex
@misc{mo-gym,
  author = {Lucas N. Alegre},
  title = {MO-Gym: Multi-Objective Reinforcement Learning Environments},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LucasAlegre/mo-gym}},
}
```

## Acknowledgments

* The `minecart-v0` env is a refactor of https://github.com/axelabels/DynMORL.
* The `deep-sea-treasure-v0` and `mo-supermario-v0` are based on https://github.com/RunzheYang/MORL.
* The `four-room-v0` is based on https://github.com/mike-gimelfarb/deep-successor-features-for-transfer.
