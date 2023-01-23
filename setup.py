from setuptools import setup


setup(name="mo-gymnasium", version="0.3.0", long_description=open("README.md").read())

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*

# https://towardsdatascience.com/create-your-own-python-package-and-publish-it-into-pypi-9306a29bc116
