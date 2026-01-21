from setuptools import find_packages, setup

setup(
    name="kmcomp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.22.0",
        "kneed>=0.7.0",
        "matplotlib>=3.2.0",
        "Pillow>=7.0.0",
    ],
    entry_points={
        "console_scripts": [
            "kmcomp = kmcomp.cli:main",
        ],
    },
)
