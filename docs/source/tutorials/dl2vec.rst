Embedding ontologies
======================

Random-walk-based embeddings
-----------------------------

This approach consists on three main steps:

* Projecting a graph out of an ontology
* Generating random walks from the graph
* Using the walks to generate the embeddings

Let's use the built-in dataset `PPIYeastSlimDataset`

.. code:: python

   from mowl.datasets.ppi_yeast import PPIYeastSlimDataset

   dataset = PPIYeastSlimDataset()


For projecting the ontology, mOWL provides different :doc:`ontology parsing methods <../api/graph/index>`. In this case let's use DL2Vec projection method, which can be imported and defined in two different ways:

The first way is calling the `DL2VecParser` class directly:

.. code:: python

   from mowl.graph.dl2vec.model import DL2VecParser
   parser = DL2VecParser(dataset.ontology, bidirectional_taxonomy = True)
   edges = parser.parse()

The second way is calling the class through a factory method:

.. code:: python
	  
   from mowl.graph.factory import parser_factory
   parser = parser_factory("dl2vec", dataset.ontology, bidirectional_taxonomy = True)
   edges = parser.parse()

The factory method will return a different parsing class depending on the first parameter.

After obtaining a graph as an edge list, we can perform random walks. As before, the walking methods can be imported directly or through a factory method. Let's use the factory method in this case.
   
.. code:: python
   
   from mowl.walking.factory import walking_factory
   walker = walking_factory("deepwalk", edges, alpha = 0.1, walk_length = 10, num_walks = 10, outfile = "data/walks")
   walker.walk()

The walks are saved in the specified filepath. After generating the walks, one usual step is to use the Word2Vec model to generate embeddings of the entities. In this case we rely on the library Gensim to use the Word2Vec model.
   
.. code:: python
   
   from gensim.models import Word2Vec
   from gensim.models.word2vec import LineSentence
   

   corpus = LineSentence("data/walks")
   
   w2v_model = Word2Vec(
	  corpus,
	  sg=1,
          min_count=1,
          vector_size=10,
          window = 10,
          epochs = 10,
          workers = 16)

   vectors = w2v_model.wv


