__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import os
import re

import gymnasium
from PIL import Image
from tqdm import tqdm

import mo_gymnasium as mo_gym


# snake to camel case: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case # noqa: E501
pattern = re.compile(r"(?<!^)(?=[A-Z])")
# how many steps to record an env for
LENGTH = 300
# iterate through all envspecs
for env_spec in tqdm(gymnasium.envs.registry.values()):
    print(env_spec.id)

    if env_spec.entry_point.split(".")[0] != "mo_gymnasium":
        continue

    try:
        env = mo_gym.make(env_spec.id, render_mode="rgb_array")
        # the gymnasium needs to be rgb renderable
        if not ("rgb_array" in env.metadata["render_modes"]):
            continue
        # extract env name/type from class path
        split = str(type(env.unwrapped)).split(".")

        # get rid of version info
        env_name = env_spec.id.split("-")[0]
        # convert NameLikeThis to name_like_this
        env_name = pattern.sub("_", env_name).lower()
        # get the env type (e.g. Box2D)
        env_type = split[2]

        pascal_env_name = env_spec.id
        snake_env_name = pattern.sub("_", pascal_env_name).lower()
        # remove what is after the last "-" in snake_env_name e.g. "-v0"
        snake_env_name = snake_env_name[: snake_env_name.rfind("-")]

        # if its an atari gymnasium
        # if env_spec.id[0:3] == "ALE":
        #     continue
        #     env_name = env_spec.id.split("-")[0][4:]
        #     env_name = pattern.sub('_', env_name).lower()

        # path for saving video
        # v_path = os.path.join("..", "pages", "environments", env_type, "videos") # noqa: E501
        # # create dir if it doesn't exist
        # if not path.isdir(v_path):
        #     mkdir(v_path)

        # obtain and save LENGTH frames worth of steps
        frames = []
        while True:
            state, info = env.reset()
            terminated, truncated = False, False
            while not (terminated or truncated) and len(frames) <= LENGTH:

                frame = env.render()
                frames.append(Image.fromarray(frame))
                action = env.action_space.sample()
                state_next, reward, terminated, truncated, info = env.step(action)

            if len(frames) > LENGTH:
                break

        env.close()

        # make sure video doesn't already exist
        # if not os.path.exists(os.path.join(v_path, env_name + ".gif")):
        frames[0].save(
            os.path.join("..", "_static", "videos", snake_env_name + ".gif"),
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
        )
        print("Saved: " + snake_env_name)

    except BaseException as e:
        print("ERROR", e)
        continue
