---
title: "Continuous Observation"
---

# Continuous Observation


MO-Gymnasium includes environments taken from the MORL literature, as well as multi-objective version of classical environments, such as Mujoco.

| Env                                                                                                                                                                                                                                                                | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`water-reservoir-v0`](https://mo-gymnasium.farama.org/environments/water-reservoir/)   <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/water-reservoir.gif" width="200px">                                | Continuous / Continuous             | `[cost_flooding, deficit_water]`                              | A Water reservoir environment. The agent executes a continuous action, corresponding to the amount of water released by the dam. From [Pianosi et al. 2013](https://iwaponline.com/jh/article/15/2/258/3425/Tree-based-fitted-Q-iteration-for-multi-objective). |
| [`mo-mountaincar-v0`](https://mo-gymnasium.farama.org/environments/mo-mountaincar/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-mountaincar.gif" width="200px">                                     | Continuous / Discrete               | `[time_penalty, reverse_penalty, forward_penalty]`            | Classic Mountain Car env, but with extra penalties for the forward and reverse actions. From [Vamplew et al. 2011](https://www.researchgate.net/publication/220343783_Empirical_evaluation_methods_for_multiobjective_reinforcement_learning_algorithms).       |
| [`mo-mountaincarcontinuous-v0`](https://mo-gymnasium.farama.org/environments/mo-mountaincarcontinuous/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-mountaincarcontinuous.gif" width="200px">       | Continuous / Continuous             | `[time_penalty, fuel_consumption_penalty]`                    | Continuous Mountain Car env, but with penalties for fuel consumption.                                                                                                                                                                                           |
| [`mo-lunar-lander-v2`](https://mo-gymnasium.farama.org/environments/mo-lunar-lander/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-lunar-lander.gif" width="200px">                                  | Continuous / Discrete or Continuous | `[landed, shaped_reward, main_engine_fuel, side_engine_fuel]` | MO version of the `LunarLander-v2` [environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/). Objectives defined similarly as in [Hung et al. 2022](https://openreview.net/forum?id=AwWaBXLIJE).                                             |
| [`minecart-v0`](https://mo-gymnasium.farama.org/environments/minecart/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/minecart.gif" width="200px">                                                       | Continuous or Image / Discrete      | `[ore1, ore2, fuel]`                                          | Agent must collect two types of ores and minimize fuel consumption. From [Abels et al. 2019](https://arxiv.org/abs/1809.07803v2).                                                                                                                               |
| [`mo-highway-v0`](https://mo-gymnasium.farama.org/environments/mo-highway/) and `mo-highway-fast-v0` <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-highway.gif" width="200px">                        | Continuous / Discrete               | `[speed, right_lane, collision]`                              | The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles and staying on the rightest lane. From [highway-env](https://github.com/eleurent/highway-env).                                                              |
| [`mo-reacher-v4`](https://mo-gymnasium.farama.org/environments/mo-reacher/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-reacher.gif" width="200px">                                                 | Continuous / Discrete               | `[target_1, target_2, target_3, target_4]`                    | Mujoco version of `mo-reacher-v0`, based on `Reacher-v4` [environment](https://gymnasium.farama.org/environments/mujoco/reacher/).                                                                                                                              |
| [`mo-hopper-v4`](https://mo-gymnasium.farama.org/environments/mo-hopper/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-hopper.gif" width="200px">                                                    | Continuous / Continuous             | `[velocity, height, energy]`                                  | Multi-objective version of [Hopper-v4](https://gymnasium.farama.org/environments/mujoco/hopper/) env.                                                                                                                                                           |
| [`mo-halfcheetah-v4`](https://mo-gymnasium.farama.org/environments/mo-halfcheetah/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-halfcheetah.gif" width="200px">                                     | Continuous / Continuous             | `[velocity, energy]`                                          | Multi-objective version of [HalfCheetah-v4](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) env. Similar to [Xu et al. 2020](https://github.com/mit-gfx/PGMORL).                                                                                |


```{toctree}
:hidden:
:glob:
:caption: MO-Gymnasium Environments

./water-reservoir
./mo-mountaincar
./mo-mountaincarcontinuous
./mo-lunar-lander
./minecart
./mo-highway
./mo-reacher
./mo-hopper
./mo-halfcheetah
```
