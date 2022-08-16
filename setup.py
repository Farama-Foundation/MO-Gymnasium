from setuptools import setup

# Environment-specific dependencies.
extras = {
    "mario": ["nes-py", "gym-super-mario-bros"],
    "minecart" : ["scipy"],
    "highway-env": ["highway-env"]
}
extras["all"] = extras["mario"] + extras["minecart"] + extras["highway-env"]


setup(
    name="mo-gym",
    version="0.1",
    description="Environments for Multi-Objective RL.",
    url="https://www.github.com/LucasAlegre/mo-gym",
    author="LucasAlegre",
    author_email="lnalegre@inf.ufrgs.br",
    license="MIT",
    packages=["mo_gym"],
    install_requires=[
        "gym==0.24.1", # 0.25 has breaking changes
        "numpy",
        "pygame",
        "scipy",
        "pymoo",
    ],
    extras_require=extras,
    tests_require=["pytest", "mock"],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

# https://towardsdatascience.com/create-your-own-python-package-and-publish-it-into-pypi-9306a29bc116