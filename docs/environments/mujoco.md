---
title: "MuJoCo"
---

# MuJoCo

Multi-objective versions of Mujoco environments.

| Env                                                                                                                                                                                                                                                                | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mo-reacher-v5`](https://mo-gymnasium.farama.org/environments/mo-reacher/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-reacher.gif" width="200px">                                                 | Continuous / Discrete               | `[target_1, target_2, target_3, target_4]`                    | Multi-objective version of `Reacher-v5` [environment](https://gymnasium.farama.org/environments/mujoco/reacher/).                                                                                                                              |
| [`mo-hopper-v5`](https://mo-gymnasium.farama.org/environments/mo-hopper/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-hopper.gif" width="200px">                                                    | Continuous / Continuous             | `[velocity, height, energy]`                                  | Multi-objective version of [Hopper-v5](https://gymnasium.farama.org/environments/mujoco/hopper/) env.                                                                                                                                                           |
| [`mo-halfcheetah-v5`](https://mo-gymnasium.farama.org/environments/mo-halfcheetah/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-halfcheetah.gif" width="200px">                                     | Continuous / Continuous             | `[velocity, energy]`                                          | Multi-objective version of [HalfCheetah-v5](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) env. Similar to [Xu et al. 2020](https://github.com/mit-gfx/PGMORL).                                                                                |
| [`mo-walker2d-v5`](https://mo-gymnasium.farama.org/environments/mo-walker2d/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-walker2d.gif" width="200px">                                              | Continuous / Continuous             | `[velocity, energy]`                                          | Multi-objective version of [Walker2d-v5](https://gymnasium.farama.org/environments/mujoco/walker2d/) env.                                                                                                                                                       |
| [`mo-ant-v5`](https://mo-gymnasium.farama.org/environments/mo-ant/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-ant.gif" width="200px">                                                             | Continuous / Continuous             | `[x_velocity, y_velocity, energy]`                            | Multi-objective version of [Ant-v5](https://gymnasium.farama.org/environments/mujoco/ant/) env.                                                                                                                                                                 |
| [`mo-swimmer-v5`](https://mo-gymnasium.farama.org/environments/mo-swimmer/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-swimmer.gif" width="200px">                                                 | Continuous / Continuous             | `[velocity, energy]`                                          | Multi-objective version of [Swimmer-v5](https://gymnasium.farama.org/environments/mujoco/swimmer/) env.                                                                                                                                                         |
| [`mo-humanoid-v5`](https://mo-gymnasium.farama.org/environments/mo-humanoid/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-humanoid.gif" width="200px">                                              | Continuous / Continuous             | `[velocity, energy]`                                          | Multi-objective version of [Humonoid-v5](https://gymnasium.farama.org/environments/mujoco/humanoid/) env.                                                                                                                                                       |


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
