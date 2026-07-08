from setuptools import setup, find_packages

setup(
    name="dotime",
    version="0.1.0",
    description="Continuous-time Causal Prior-Fitted Networks: in-context causal "
                "effect estimation for continuous-time temporal data",
    # Discovers the ``dotime`` package plus the vendored ``causal_time_prior``
    # and ``dopfnprior`` packages at the repository root (self-contained; no
    # external repositories or PYTHONPATH manipulation required).
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "scipy",
        "pandas",
        "networkx",         # vendored causal_time_prior graph sampling
        "pfns",             # PFN bar-distribution head (pfns.model.bar_distribution)
        "matplotlib",
        "pyyaml",
        "causalchamber",    # Causal Chamber real-data benchmark
        "flash-linear-attention",
        "wandb",
    ],
    extras_require={
        "analysis": ["statsmodels"],  # VAR / AR baselines
    },
)
