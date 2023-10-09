<p align="center">
  <img src= "https://github.com/bio-ontology-research-group/mowl/blob/main/docs/source/mowl_black_background_colors_2048x2048px.png?raw=true" width="300"/>
</p>
  
# mOWL: Machine Learning Library with Ontologies

**mOWL** is a library that provides different machine learning methods in which ontologies are used as background knowledge. **mOWL** is developed 
mainly in Python, but we have integrated the functionalities of [OWLAPI](https://github.com/owlcs/owlapi), which is written in Java, for which we use [JPype](https://jpype.readthedocs.io/en/latest/) to bind Python with the Java Virtual Machine (JVM).


## Table of contents
  - [Installation](#installation)
  - [List of contributors](#list-of-contributors)
  - [License](#license)
  - [Documentation](#documentation)
  - [Changelog](#changelog)


## Installation

### System dependencies

  - JDK version 17
  - Python version: 3.8, 3.9, 3.10, 3.11
  - Conda version >= 4.x.x

### Python requirements

  - Gensim >= 4.x.x
  - PyTorch >= 1.12.x
  - PyKEEN >= 1.10.1

### Install from PyPi

```
pip install mowl-borg
```

### Build from source
Installation can be done with the following commands:

```
git clone https://github.com/bio-ontology-research-group/mowl.git

cd mowl

conda env create -f envs/environment_3.8.yml
conda activate mowl

./build_jars.sh

python setup.py install
```

The line `./build_jars.sh` will generate the necessary `jar` files to bind Python with the code that runs in the JVM. After building, a ``.tar.gz`` file will be generated under `dist` and can be used to install mOWL.



## List of contributors
* [Fernando Zhapa](https://github.com/ferzcam)
* [Maxat Kulmanov](https://github.com/coolmaksat)
* [Sarah Alghamdi](https://github.com/smalghamdi)
* [Robert Hoehndorf](https://github.com/leechuck)
* [Carsten Jahn](https://github.com/carsten-jahn)
* [Sonja Katz](https://github.com/sonjakatz)
* [Marco Anteghini](https://github.com/MarcoAnteghini)
* Francesco Gualdi
* [luis-sribeiro](https://github.com/luis-sribeiro)
* [Leduin Cuenca](https://github.com/leduin) (logo)

## License
This software library is distributed under the [BSD-3-Clause license](https://github.com/bio-ontology-research-group/mowl/blob/main/LICENSE)

## Documentation
Full documentation and API reference can be found in our [ReadTheDocs](https://mowl.readthedocs.io/en/latest/index.html) website.

## ChangeLog
ChangeLog is available in our [changelog file](https://github.com/bio-ontology-research-group/mowl/blob/main/CHANGELOG.md) and also in the [release section](https://github.com/bio-ontology-research-group/mowl/releases/).

## Citation
If you used mOWL in your work, please consider citing [this article](https://doi.org/10.1093/bioinformatics/btac811):

```bibtex
@article{10.1093/bioinformatics/btac811,
    author = {Zhapa-Camacho, Fernando and Kulmanov, Maxat and Hoehndorf, Robert},
    title = "{mOWL: Python library for machine learning with biomedical ontologies}",
    journal = {Bioinformatics},
    year = {2022},
    month = {12},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac811},
    url = {https://doi.org/10.1093/bioinformatics/btac811},
    note = {btac811},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btac811/48438324/btac811.pdf},
}
```
