---
title: "Discrete Observation"
---

# Discrete Observation

Environments with discrete observation spaces.

| Env                                                                                                                                                                                                                                                                | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deep-sea-treasure-v0`](https://mo-gymnasium.farama.org/environments/deep-sea-treasure/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/deep-sea-treasure.gif" width="200px">                            | Discrete / Discrete                 | `[treasure, time_penalty]`                                    | Agent is a submarine that must collect a treasure while taking into account a time penalty. Treasures values taken from [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf).                                                                               |
| [`deep-sea-treasure-concave-v0`](https://mo-gymnasium.farama.org/environments/deep-sea-treasure-concave/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/deep-sea-treasure-concave.gif" width="200px">    | Discrete / Discrete                 | `[treasure, time_penalty]`                                    | Agent is a submarine that must collect a treasure while taking into account a time penalty. Treasures values taken from [Vamplew et al. 2010](https://link.springer.com/article/10.1007/s10994-010-5232-5).                                                     |
| [`deep-sea-treasure-mirrored-v0`](https://mo-gymnasium.farama.org/environments/deep-sea-treasure-mirrored/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/deep-sea-treasure-mirrored.gif" width="200px"> | Discrete / Discrete                 | `[treasure, time_penalty]`                                    | Harder version of the concave DST [Felten et al. 2022](https://www.scitepress.org/Papers/2022/109891/109891.pdf).                                                                                                                                            |
| [`resource-gathering-v0`](https://mo-gymnasium.farama.org/environments/resource-gathering/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/resource-gathering.gif" width="200px">                         | Discrete / Discrete                 | `[enemy, gold, gem]`                                          | Agent must collect gold or gem. Enemies have a 10% chance of killing the agent. From [Barret & Narayanan 2008](https://dl.acm.org/doi/10.1145/1390156.1390162).                                                                                                 |
| [`fishwood-v0`](https://mo-gymnasium.farama.org/environments/fishwood/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/screenshots/fishwood.png" width="200px">                                                  | Discrete / Discrete                 | `[fish_amount, wood_amount]`                                  | ESR environment, the agent must collect fish and wood to light a fire and eat. From [Roijers et al. 2018](https://www.researchgate.net/publication/328718263_Multi-objective_Reinforcement_Learning_for_the_Expected_Utility_of_the_Return).                    |
| [`breakable-bottles-v0`](https://mo-gymnasium.farama.org/environments/breakable-bottles/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/breakable-bottles.gif" width="200px">                            | Discrete (Dictionary) / Discrete    | `[time_penalty, bottles_delivered, potential]`                | Gridworld with 5 cells. The agents must collect bottles from the source location and deliver to the destination. From [Vamplew et al. 2021](https://www.sciencedirect.com/science/article/pii/S0952197621000336).                                               |
| [`fruit-tree-v0`](https://mo-gymnasium.farama.org/environments/fruit-tree/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/fruit-tree.gif" width="200px">                                            | Discrete / Discrete                 | `[nutri1, ..., nutri6]`                                       | Full binary tree of depth d=5,6 or 7. Every leaf contains a fruit with a value for the nutrients Protein, Carbs, Fats, Vitamins, Minerals and Water. From [Yang et al. 2019](https://arxiv.org/pdf/1908.08342.pdf).                                             |
| [`four-room-v0`](https://mo-gymnasium.farama.org/environments/four-room/) <br><img src="https://raw.githubusercontent.com/Farama-Foundation/MO-Gymnasium/main/docs/_static/videos/four-room.gif" width="200px">                                                    | Discrete / Discrete                 | `[item1, item2, item3]`                                       | Agent must collect three different types of items in the map and reach the goal. From [Alegre et al. 2022](https://proceedings.mlr.press/v162/alegre22a.html).                                                                                                  |


```{toctree}
:hidden:
:glob:
:caption: MO-Gymnasium Environments

./deep-sea-treasure
./deep-sea-treasure-concave
./deep-sea-treasure-mirrored
./resource-gathering
./four-room
./fruit-tree
./breakable-bottles
./fishwood


```
