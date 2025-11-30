"""
Setup script for the EEG-Kuramoto Neural Dynamics Framework.
"""

from setuptools import setup, find_packages

setup(
    name="cerebral_flow",
    version="0.1.0",
    description="CerebralFlow: Neural Dynamics Simulation Framework",
    author="CerebralFlow Team",
    author_email="contact@cerebralflow.org",
    url="https://github.com/cerebralflow/cerebralflow",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.23.0",
        "networkx>=2.5.0",
        "pandas>=1.1.0",
        "mne>=0.20.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
)