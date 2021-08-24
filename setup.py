# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:01:22 2020

@author: Victor
"""
import setuptools
print(setuptools.find_namespace_packages(where='pyparax'))

setuptools.setup(
    name="pyparax",
    version="5.0",
    author="Victor-Cristian Palea",
    author_email="victorcristianpalea@gmail.com",
    description="Optical simulator for paraxial approximation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)