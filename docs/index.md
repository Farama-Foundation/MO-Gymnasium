---
hide-toc: true
firstpage:
lastpage:
---

```{toctree}
:hidden:
:caption: API

api/api
```

```{toctree}
:hidden:
:caption: Environments

environments/environments
```

```{toctree}
:hidden:
:caption: Examples

examples/morl_baselines
```

```{toctree}
:hidden:
:caption: Wrappers

wrappers/wrappers
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/MO-Gymnasium>
Donate <https://farama.org/donations>

```

# MO-Gymnasium is a standardized API and a suite of environments for multi-objective reinforcement learning (MORL)

For details on multi-objective MDP's (MOMDP's) and other MORL definitions, see [A practical guide to multi-objective reinforcement learning and planning](https://link.springer.com/article/10.1007/s10458-022-09552-y).

## Install

### From Pypi
```bash
pip install mo-gymnasium
```

### From source
```bash
git clone https://github.com/Farama-Foundation/MO-Gymnasium
cd MO-Gymnasium
pip install -e .
```

## Citing
If you use this repository in your work, please cite:

```bibtex
@inproceedings{Alegre+2022bnaic,
  author = {Lucas N. Alegre and Florian	Felten and El-Ghazali Talbi and Gr{\'e}goire Danoy and Ann Now{\'e} and Ana L. C. Bazzan and Bruno C. da Silva},
  title = {{MO-Gym}: A Library of Multi-Objective Reinforcement Learning Environments},
  booktitle = {Proceedings of the 34th Benelux Conference on Artificial Intelligence BNAIC/Benelearn 2022},
  year = {2022}
}
```
