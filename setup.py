import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mowl-borg",
    version="0.0.28",
    author="Bio-Ontology Research Group",
    author_email="fernando.zhapacamacho@kaust.edu.sa",
    description="mOWL: A machine learning library with ontologies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bio-ontology-research-group/mowl",
    project_urls={
        "Bug Tracker": "https://github.com/bio-ontology-research-group/mowl/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where=".", exclude=("tests",)),
    package_data={"mowl": ["lib/*.jar"]},
    python_requires=">=3.8",
    install_requires=[
        "JPype1==1.3.0",
        "numpy",
        "networkx",
        "click",
        "pandas",
        "scipy",
        "pyyaml",
        "scikit-learn",
        "urllib3",
        "torch",
        "gensim",
        "requests",
    ]
)

