---
title: "Image Observation"
---

# Image Observation


MO-Gymnasium environments with image observation spaces.

| Env                                                                                                                                                                                                                                                                | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`minecart-rgb-v0`](https://mo-gymnasium.farama.org/environments/minecart/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/minecart.gif" width="200px">                                                       | Continuous or Image / Discrete      | `[ore1, ore2, fuel]`                                          | Agent must collect two types of ores and minimize fuel consumption. From [Abels et al. 2019](https://arxiv.org/abs/1809.07803v2).                                                                                                                               |
| [`mo-supermario-v0`](https://mo-gymnasium.farama.org/environments/mo-supermario/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-supermario.gif" width="200px">                                        | Image / Discrete                    | `[x_pos, time, death, coin, enemy]`                           | [:warning: SuperMarioBrosEnv support is limited.] Multi-objective version of [SuperMarioBrosEnv](https://github.com/Kautenja/gym-super-mario-bros). Objectives are defined similarly as in [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf).            |

```{toctree}
:hidden:
:glob:
:caption: MO-Gymnasium Environments

./minecart-rgb
./mo-supermario

```
