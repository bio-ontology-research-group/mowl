<p align="center">
  <img src="docs/source/mowl_black_background_colors_2048x2048px.png" width="300"/>
</p>
  
# mOWL: Machine Learning Library with Ontologies

**mOWL** is a library that provides different machine learning methods in which ontologies are used as background knowledge. **mOWL** is developed 
mainly in Python, but we have integrated the functionalities of [OWLAPI](https://github.com/owlcs/owlapi), which is written in Java, for which we use [JPype](https://jpype.readthedocs.io/en/latest/) to bind Python with the Java Virtual Machine (JVM).


## Table of contents
  - [Installation](#installation)
  - [Examples of use](#examples-of-use)
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

## Examples of use

### Basic example

In this example we use the training data (which is an OWL ontology) from the built-in dataset [PPIYeasSlimDataset](https://mowl.readthedocs.io/en/latest/api/datasets/index.html#mowl.datasets.ppi_yeast.PPIYeastSlimDataset) to build a graph representation using the _subClassOf_ axioms.

```python
from mowl.datasets.ppi_yeast import PPIYeastSlimDataset
from mowl.graph.taxonomy.model import TaxonomyParser

dataset = PPIYeastSlimDataset()
parser = TaxonomyParser(dataset.ontology, bidirectional_taxonomy = True)
edges = parser.parse()
```
The projected `edges` is an edge list of a graph. One use of this may be to generate random walks:

```python
from mowl.walking.deepwalk.model import DeepWalk
walker = DeepWalk(edges,
	              100, # number of walks
				  20, # length of each walk
				  0.2, # probability of restart
				  workers = 4, # number of usable CPUs
				  )

walker.walk()
walks = walker.walks
```

### Ontology to graph

In the previous example we called the class `TaxonomyParser` to perform the graph projection. However, there are more ways to perform the projection. We include the following four:

* [TaxonomyParser](https://mowl.readthedocs.io/en/latest/api/graph/index.html#subclass-hierarchy): "taxonomy"
* [TaxonomyWithRelsParser](https://mowl.readthedocs.io/en/latest/api/graph/index.html#subclass-hierarchy-with-relations): "taxonomy_rels"
* [DL2VecParser](https://mowl.readthedocs.io/en/latest/api/graph/index.html#dl2vec-graph): "dl2vec"
* [OWL2VecParser](https://mowl.readthedocs.io/en/latest/api/graph/index.html#dl2vec-graph): "owl2vec_star"

Instead of instantianting each of them separately, there is the following _factory_ method:
```python
from mowl.graph.factory import parser_factory

parser = parser_factory("taxonomy_rels", dataset.ontology, bidirectional_taxonomy = True)
```
Now `parser` will be an instance of the `TaxonomyWithRelsParser` class. The string parameters for each method are listed above.

For the random walks method we have a similar factory method that can be found in `mowl.walking.factory` and is called `walking_factory`.


## List of contributors

## License

## Docker

```bash
docker build --tag mowl .
docker
docker run --mount type=bind,source="$(pwd)"/,target=/home/mowl -it -p 8888:8888 --name mowl mowlconda
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```

## Documentation

Full documentation and API reference can be found in our [ReadTheDocs](https://mowl.readthedocs.io/en/latest/index.html) website.
