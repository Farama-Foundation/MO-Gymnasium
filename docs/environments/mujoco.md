---
title: "MuJoCo"
---

# MuJoCo

Multi-objective versions of Mujoco environments.

| Env                                                                                                                                                                                                                                                                | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mo-reacher-v4`](https://mo-gymnasium.farama.org/environments/mo-reacher/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-reacher.gif" width="200px">                                                 | Continuous / Discrete               | `[target_1, target_2, target_3, target_4]`                    | Mujoco version of `mo-reacher-v0`, based on `Reacher-v4` [environment](https://gymnasium.farama.org/environments/mujoco/reacher/).                                                                                                                              |
| [`mo-hopper-v4`](https://mo-gymnasium.farama.org/environments/mo-hopper/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-hopper.gif" width="200px">                                                    | Continuous / Continuous             | `[velocity, height, energy]`                                  | Multi-objective version of [Hopper-v4](https://gymnasium.farama.org/environments/mujoco/hopper/) env.                                                                                                                                                           |
| [`mo-halfcheetah-v4`](https://mo-gymnasium.farama.org/environments/mo-halfcheetah/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-halfcheetah.gif" width="200px">                                     | Continuous / Continuous             | `[velocity, energy]`                                          | Multi-objective version of [HalfCheetah-v4](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) env. Similar to [Xu et al. 2020](https://github.com/mit-gfx/PGMORL).                                                                                |
| [`mo-walker2d-v4`](https://mo-gymnasium.farama.org/environments/mo-walker2d/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-walker2d.gif" width="200px">                                              | Continuous / Continuous             | `[velocity, energy]`                                          | Multi-objective version of [Walker2d-v4](https://gymnasium.farama.org/environments/mujoco/walker2d/) env.                                                                                                                                                       |
| [`mo-ant-v4`](https://mo-gymnasium.farama.org/environments/mo-ant/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-ant.gif" width="200px">                                                             | Continuous / Continuous             | `[x_velocity, y_velocity]`                                    | Multi-objective version of [Ant-v4](https://gymnasium.farama.org/environments/mujoco/ant/) env.                                                                                                                                                                 |
| [`mo-swimmer-v4`](https://mo-gymnasium.farama.org/environments/mo-swimmer/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-swimmer.gif" width="200px">                                                 | Continuous / Continuous             | `[velocity, energy]`                                          | Multi-objective version of [Swimmer-v4](https://gymnasium.farama.org/environments/mujoco/swimmer/) env.                                                                                                                                                         |
| [`mo-humanoid-v4`](https://mo-gymnasium.farama.org/environments/mo-humanoid/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-humanoid.gif" width="200px">                                              | Continuous / Continuous             | `[velocity, energy]`                                          | Multi-objective version of [Humonoid-v4](https://gymnasium.farama.org/environments/mujoco/humanoid/) env.                                                                                                                                                       |


```{toctree}
:hidden:
:glob:
:caption: MO-Gymnasium Environments

./mo-reacher
./mo-hopper
./mo-halfcheetah
./mo-walker2d
./mo-ant
./mo-swimmer
./mo-humanoid
```
