from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="causationentropy",
    version="0.1.0",
    author="Kevin Slote",  # Replace with your name
    author_email="kslote1@gmail.com",  # Replace with your email
    description="Causal network discovery using optimal causation entropy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kslote1/causationentropy",  # Replace with your repo URL
    project_urls={
        "Bug Tracker": "https://github.com/kslote1/causationentropy/issues",
        "Documentation": "https://github.com/kslote1/causationentropy",
        "Source Code": "https://github.com/kslote1/causationentropy",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "numpydoc>=1.0",
        ],
        "plotting": [
            "seaborn>=0.11",
            "plotly>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "causationentropy-cli=causationentropy.cli:main",  # Optional CLI interface
        ],
    },
    keywords="causality, entropy, time-series, network, causal-discovery, information-theory",
    include_package_data=True,
    zip_safe=False,
)
