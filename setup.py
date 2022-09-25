from setuptools import setup, find_packages

# Environment-specific dependencies.
extras = {
    "mario": ["nes-py", "gym-super-mario-bros"],
    "minecart" : ["scipy"]
}
extras["all"] = extras["mario"] + extras["minecart"]

packages = [package for package in find_packages() if package.startswith("mo_gym")]
package_data = {package: ["*.json", "assets/*"] for package in packages}

setup(
    name="mo-gym",
    version="0.1.2",
    description="Environments for Multi-Objective RL.",
    url="https://www.github.com/LucasAlegre/mo-gym",
    author="LucasAlegre",
    author_email="lnalegre@inf.ufrgs.br",
    license="MIT",
    packages=packages,
    package_data=package_data,
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