#!/usr/bin/env python

from os import path
from setuptools import setup, find_packages


test_glob_patterns = ["*.tests", "*.tests.*", "tests.*", "tests"]

setup(
    name="phantom",
    version="0.1",
    description="Multi-agent simulator for OTC markets",
    url="https://us-east-2.console.aws.amazon.com/codesuite/codecommit/repositories/phantom-core/setup",
    author="JPM AI Research",
    scripts=["scripts/phantom"],
    packages=find_packages(exclude=test_glob_patterns),
    # install_requires=[x.strip() for x in
    #                     open(path.join(path.abspath(path.dirname(__file__)), 'requirements.txt')).readlines()],
)
