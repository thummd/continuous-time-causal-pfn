from setuptools import setup, find_packages

setup(
    name="dotime",
    version="0.1.0",
    description="Do-Over-Time-PFN: In-context causal effect estimation for temporal data",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "matplotlib",
        "pyyaml",
        "causalchamber",
        "flash-linear-attention",
        "wandb",
    ],
    extras_require={
        "analysis": ["statsmodels"],
    },
)
