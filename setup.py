from setuptools import find_packages, setup

setup(
    name="banqi-training",
    version="0.1.0",
    description="Banqi 分布式训练器（被 banqi-scheduler 调度）",
    packages=find_packages(),
    package_data={"banqi_training": ["config.default.yaml"]},
    python_requires=">=3.10",
)
