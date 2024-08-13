---
title: "Classic Control"
---

# Classic Control

Multi-objective versions of classical Gymnasium's environments.

| Env                                                                                                                                                                                                                                                                | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mo-mountaincar-v0`](https://mo-gymnasium.farama.org/environments/mo-mountaincar/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-mountaincar.gif" width="200px">                                     | Continuous / Discrete               | `[time_penalty, reverse_penalty, forward_penalty]`            | Classic Mountain Car env, but with extra penalties for the forward and reverse actions. From [Vamplew et al. 2011](https://www.researchgate.net/publication/220343783_Empirical_evaluation_methods_for_multiobjective_reinforcement_learning_algorithms).       |
[`mo-mountaincar-3d-v0`] ** <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-mountaincar.gif" width="200px">  | Continuous / Discrete| `[time_penalty, move_penalty, speed_objective]` | The forward and backward penalties have been merged into the move penalty and a speed objective has been introduced which gives the positive reward equivalent to the car's speed at that time step.* |
[`mo-mountaincar-timemove-v0`] ** <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-mountaincar.gif" width="200px">  | Continuous / Discrete | `[time_penalty, move_penalty]`| Class Mountain Car env but an extra penalty for moving backwards or forwards merged into a move penalty. |
[`mo-mountaincar-timespeed-v0`] ** <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-mountaincar.gif" width="200px"> | Continuous / Discrete| `[time_penalty, speed_objective]` | Class Mountain Car env but an extra positive objective of speed which gives the positive reward equivalent to the car's speed at that time step.*
| [`mo-mountaincarcontinuous-v0`](https://mo-gymnasium.farama.org/environments/mo-mountaincarcontinuous/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-mountaincarcontinuous.gif" width="200px">       | Continuous / Continuous             | `[time_penalty, fuel_consumption_penalty]`                    | Continuous Mountain Car env, but with penalties for fuel consumption.                                                                                                                     |
| [`mo-lunar-lander-v2`](https://mo-gymnasium.farama.org/environments/mo-lunar-lander/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-lunar-lander.gif" width="200px">                                  | Continuous / Discrete or Continuous | `[landed, shaped_reward, main_engine_fuel, side_engine_fuel]` | MO version of the `LunarLander-v2` [environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/). Objectives defined similarly as in [Hung et al. 2022](https://openreview.net/forum?id=AwWaBXLIJE).                                             |

*An additional objective was introduced to prevent the agent from converging to the local maxima due to a lack of reward signal for the static action. 

**Read more about these environments and the detailed reasoning behind them in [`Pranav Gupta's Dissertation`](https://drive.google.com/file/d/1yT6hlavYZGmoB2phaIBX_5hbibA3Illa/view?usp=sharing)

```{toctree}
:hidden:
:glob:
:caption: MO-Gymnasium Environments

./mo-mountaincar
./mo-mountaincarcontinuous
./mo-lunar-lander
./mo-lunar-lander-continuous

```
