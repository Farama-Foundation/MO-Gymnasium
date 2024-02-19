---
title: "Classic Control"
---

# Classic Control


MO-Gymnasium includes environments taken from the MORL literature, as well as multi-objective version of classical environments.

| Env                                                                                                                                                                                                                                                                | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mo-mountaincar-v0`](https://mo-gymnasium.farama.org/environments/mo-mountaincar/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-mountaincar.gif" width="200px">                                     | Continuous / Discrete               | `[time_penalty, reverse_penalty, forward_penalty]`            | Classic Mountain Car env, but with extra penalties for the forward and reverse actions. From [Vamplew et al. 2011](https://www.researchgate.net/publication/220343783_Empirical_evaluation_methods_for_multiobjective_reinforcement_learning_algorithms).       |
| [`mo-mountaincarcontinuous-v0`](https://mo-gymnasium.farama.org/environments/mo-mountaincarcontinuous/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-mountaincarcontinuous.gif" width="200px">       | Continuous / Continuous             | `[time_penalty, fuel_consumption_penalty]`                    | Continuous Mountain Car env, but with penalties for fuel consumption.                                                                                                                                                                                           |
| [`mo-lunar-lander-v2`](https://mo-gymnasium.farama.org/environments/mo-lunar-lander/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/mo-lunar-lander.gif" width="200px">                                  | Continuous / Discrete or Continuous | `[landed, shaped_reward, main_engine_fuel, side_engine_fuel]` | MO version of the `LunarLander-v2` [environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/). Objectives defined similarly as in [Hung et al. 2022](https://openreview.net/forum?id=AwWaBXLIJE).                                             |

```{toctree}
:hidden:
:glob:
:caption: MO-Gymnasium Environments

./mo-mountaincar
./mo-mountaincarcontinuous
./mo-lunar-lander

```
