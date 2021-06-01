from setuptools import find_namespace_packages, setup

setup(
    name="goddard_rl",
    version="0.1.0",
    description="CS 295 Reinforcement Learning: Solving Goddard Problem using RL",
    author="James-Andrew Sarmiento",
    packages=find_namespace_packages(include=["src*", "gym_goddard*"]),
    python_requires="~=3.7.1",
    install_requires=[
        "gym~=0.18.3",
        "torch~=1.8.1",
        "tensorflow~=1.14.0",
        "matplotlib~=3.3.4",
        "wandb~=0.10.31",
        "gdown~=3.13",
        "numpy~=1.16.4"
    ],
)