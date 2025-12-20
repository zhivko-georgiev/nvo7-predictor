"""Setup configuration for NVO Rankings package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nvo-rankings",
    version="1.0.0",
    author="Zhivko Georgiev",
    author_email="zhivko.d.georgiev@gmail.com",
    description="ML-based prediction system for Bulgarian 7th grade high school admission rankings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "pandas>=2.0.0,<3.0.0",
        "numpy>=1.24.0,<2.0.0",
        "xgboost>=2.0.0,<3.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "openpyxl>=3.1.0,<4.0.0",
        "PyYAML>=6.0,<7.0",
        "click>=8.1.0,<9.0.0",
        "tqdm>=4.65.0,<5.0.0",
    ],
    entry_points={
        "console_scripts": [
            "nvo=nvo.cli.commands:cli",
        ],
    },
)
