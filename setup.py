from setuptools import find_packages, setup

setup(
    name="srs_sampler",
    version="0.1.0",
    description="PyTorch Dataset Sampler inspired by Spaced Repitition Systems like Anki",
    author="Dustin Ewan",
    url="https://github.com/dustinewan/spaced-repetition-sampler",
    packages=find_packages(),
    install_requires=["torch"],
)
