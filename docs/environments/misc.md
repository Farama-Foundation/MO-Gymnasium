---
title: "Misc"
---

# Miscellaneous

MO-Gymnasium includes environments taken from the MORL literature, as well as multi-objective version of classical environments, such as Mujoco.

| Env                                                                                                                                                                                                                                                                | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`water-reservoir-v0`](https://mo-gymnasium.farama.org/environments/water-reservoir/)   <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/water-reservoir.gif" width="200px">                                | Continuous / Continuous             | `[cost_flooding, deficit_water]`                              | A Water reservoir environment. The agent executes a continuous action, corresponding to the amount of water released by the dam. From [Pianosi et al. 2013](https://iwaponline.com/jh/article/15/2/258/3425/Tree-based-fitted-Q-iteration-for-multi-objective). |
| [`minecart-v0`](https://mo-gymnasium.farama.org/environments/minecart/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/minecart.gif" width="200px">                                                       | Continuous or Image / Discrete      | `[ore1, ore2, fuel]`                                          | Agent must collect two types of ores and minimize fuel consumption. From [Abels et al. 2019](https://arxiv.org/abs/1809.07803v2).                                                                                                                               |
| [`mo-highway-v0`](https://mo-gymnasium.farama.org/environments/mo-highway/) and `mo-highway-fast-v0` <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-highway.gif" width="200px">                        | Continuous / Discrete               | `[speed, right_lane, collision]`                              | The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles and staying on the rightest lane. From [highway-env](https://github.com/eleurent/highway-env).                                                              |
| [`mo-supermario-v0`](https://mo-gymnasium.farama.org/environments/mo-supermario/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-supermario.gif" width="200px">                                        | Image / Discrete                    | `[x_pos, time, death, coin, enemy]`                           | [:warning: SuperMarioBrosEnv support is limited.] Multi-objective version of [SuperMarioBrosEnv](https://github.com/Kautenja/gym-super-mario-bros). Objectives are defined similarly as in [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf).            |


```{toctree}
:hidden:
:glob:
:caption: MO-Gymnasium Environments

./water-reservoir
./minecart
./mo-highway
./mo-supermario
```
