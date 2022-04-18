from setuptools import find_packages, setup

# Environment-specific dependencies.
extras = {
    "mario": ["nes-py", "gym-super-mario-bros"],
    "minecart" : ["scipy"]
}
extras["all"] = extras["mario"] + extras["minecart"]


setup(
    name="mo-gym",
    version="0.1",
    description="MO-Gym: Environments for Multi-Objective RL.",
    url="https://www.github.com/LucasAlegre/mo-gym",
    author="LucasAlegre",
    author_email="lnalegre@inf.ufrgs.br",
    license="MIT",
    packages=["mo_gym"],
    install_requires=[
        "gym",
        "numpy",
        "pygame",
        "scipy"
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