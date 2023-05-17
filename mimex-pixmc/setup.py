#!/usr/bin/env python3

"""Setup MIMEx."""

from setuptools import find_packages, setup


setup(
    name="mimex",
    version="0.1.0",
    description="MIMEx: Intrinsic Rewards from Masked Input Modeling",
    packages=find_packages(),
    python_requires=">=3.6.*",
    install_requires=[
        "iopath",
        "timm",
    ],
)
