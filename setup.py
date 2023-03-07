import pathlib

from setuptools import setup


CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the mo-gymnasium version."""
    path = CWD / "mo_gymnasium" / "__init__.py"
    content = path.read_text()
    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(name="mo-gymnasium", version=get_version(), long_description=open("README.md").read())

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*

# https://towardsdatascience.com/create-your-own-python-package-and-publish-it-into-pypi-9306a29bc116
