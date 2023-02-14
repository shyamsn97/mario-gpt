import io
import os
import re
from os import path

from setuptools import find_packages
from setuptools import setup


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="mario-gpt",
    version="0.1.0",
    url="https://github.com/kragniz/cookiecutter-pypackage-minimal",
    license='MIT',

    author="Shyam Sudhakaran",
    author_email="shyamsnair@protonmail.com",

    description="Generating Mario Levels with GPT2. Code for the paper: 'MarioGPT: Open-Ended Text2Level Generation through Large Language Models', https://arxiv.org/abs/2302.05981",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        'torch',
        'transformers',
        'scipy',
        'tqdm'
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
