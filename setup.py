from setuptools import setup, find_packages
from io import open
from os import path

import pathlib
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# automatically captured required modules for install_requires in requirements.txt
with open(path.join(HERE, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if ('git+' not in x) and (
    not x.startswith('#')) and (not x.startswith('-'))]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs \
                    if 'git+' not in x]


setup(
    name="geoshapley",
    version='0.0.3',
    description="GeoShapley value for measuring spatial effects",
    long_description="GeoShapley: A game theory based approach to measuring spatial effects from machine learning models",
    author="Ziqi Li",
    author_email="liziqi1992@gmail.com",
    url="https://github.com/Ziqi-Li/geoshapley",
    license="MIT",
    entry_points={'console_scripts': ['geoshapley=geoshapley.__main__:main',],},
    dependency_links=dependency_links,
    install_requires = install_requires,
    packages = find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ]
)
