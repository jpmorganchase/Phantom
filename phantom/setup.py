#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import re

TEST_GLOB_PATTERNS = ["*.tests", "*.tests.*", "tests.*", "tests"]

NAME = "phantom"


def _get_version():
    with open("../version.txt") as fp:
        return fp.readlines()[0]


def _get_long_description():
    with open(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
        encoding="utf-8",
    ) as readme_file:
        long_description = readme_file.read()
    return long_description


def _get_requirements():
    with open(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "requirements.txt"),
        encoding="utf-8",
    ) as requirements_file:
        requirements = [
            l.strip()
            for l in requirements_file.readlines()
            if not (l.strip().startswith("#") or l.strip().startswith("-"))
        ]
    return requirements


setup(
    name=NAME,
    version=_get_version(),
    description="A Multi-agent reinforcement-learning simulator framework.",
    long_description=_get_long_description(),
    python_requires=">3.7.0",
    url="https://github.com/jpmorganchase/Phantom",
    author="JPM AI Research",
    classifiers=[
        "Development Status :: 5 - Production",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    keywords="ai research reinforcement learning simulator multi-agent",
    packages=find_packages(exclude=TEST_GLOB_PATTERNS),
    install_requires=_get_requirements(),
)
