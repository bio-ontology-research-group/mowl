import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mowl-borg",
    version="1.0.2",
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
    python_requires=">=3.9",
    install_requires=[
        "class-resolver==0.5.4",
        "click",
        "deprecated",
        "gensim>=4.3.0",
        "JPype1==1.5.1",
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "pykeen==1.11.0",
        "pyyaml",
        "requests",
        "scikit-learn",
        "scipy<1.16.0",
        "temp",
        "tqdm",
        "torch",
        "urllib3"
    ],

)
