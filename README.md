<p align="center">
  <img src="docs/source/mowl_black_background_colors_2048x2048px.png" width="300"/>
</p>
  
# mOWL: Machine Learning Library with Ontologies

**mOWL** is a library that provides different machine learning methods in which ontologies are used as background knowledge. **mOWL** is developed 
mainly in Python, but we have integrated the functionalities of [OWLAPI](https://github.com/owlcs/owlapi), which is written in Java, for which we use [JPype](https://jpype.readthedocs.io/en/latest/) to bind Python with the Java Virtual Machine (JVM).


## Table of contents
  - [Installation](#installation)
  - [Usage](#usage)
  - [List of contributors](#list-of-contributors)


## Installation

Installation can be done with the following commands:

```
git clone https://github.com/bio-ontology-research-group/mowl.git

cd mowl

conda env create -f environment.yml
conda activate mowl

cd mowl
./rebuild.sh
```

The last line will generate the necessary `jar` files to bind Python with the code that runs in the JVM

## Usage

## List of contributors

## License

## Documentation

Full documentation and API reference can be found in our [ReadTheDocs](https://mowl.readthedocs.io/en/latest/index.html) website.
