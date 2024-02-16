---
title: "Mujoco"
---

# Mujoco

Multi-objective versions of Mujoco environments.

| Env                                                                                                                                                                                                                                                                | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mo-reacher-v4`](https://mo-gymnasium.farama.org/environments/mo-reacher/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-reacher.gif" width="200px">                                                 | Continuous / Discrete               | `[target_1, target_2, target_3, target_4]`                    | Mujoco version of `mo-reacher-v0`, based on `Reacher-v4` [environment](https://gymnasium.farama.org/environments/mujoco/reacher/).                                                                                                                              |
| [`mo-hopper-v4`](https://mo-gymnasium.farama.org/environments/mo-hopper/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-hopper.gif" width="200px">                                                    | Continuous / Continuous             | `[velocity, height, energy]`                                  | Multi-objective version of [Hopper-v4](https://gymnasium.farama.org/environments/mujoco/hopper/) env.                                                                                                                                                           |
| [`mo-halfcheetah-v4`](https://mo-gymnasium.farama.org/environments/mo-halfcheetah/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-halfcheetah.gif" width="200px">                                     | Continuous / Continuous             | `[velocity, energy]`                                          | Multi-objective version of [HalfCheetah-v4](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) env. Similar to [Xu et al. 2020](https://github.com/mit-gfx/PGMORL).                                                                                |


```{toctree}
:hidden:
:glob:
:caption: MO-Gymnasium Environments

./mujoco/*

```
