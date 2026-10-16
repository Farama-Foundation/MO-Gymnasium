---
title: "Vector Wrappers"
---

# Vector Wrappers

Similar to the normal wrappers, MO-Gymnasium provides a few wrappers that are specifically designed to work with vectorized environments. They are all available directly from the `mo_gymnasium.wrappers.vector` module.


## `MOSyncVectorEnv`

```{eval-rst}
.. autoclass:: mo_gymnasium.wrappers.vector.MOSyncVectorEnv
```

## `MORecordEpisodeStatistics`

Episode statistics include every transition when using `AutoresetMode.SAME_STEP`.
With `AutoresetMode.NEXT_STEP`, the reset-only step after an episode ends is not
included in the next episode's return or length.

```{eval-rst}
.. autoclass:: mo_gymnasium.wrappers.vector.MORecordEpisodeStatistics
```
