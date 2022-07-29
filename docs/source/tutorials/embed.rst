Embedding ontologies
======================

Random-walk-based embeddings
-----------------------------

This approach consists on three main steps:

* Projecting a graph out of an ontology
* Generating random walks from the graph
* Using the walks to generate the embeddings

Let's use the built-in dataset ``PPIYeastSlimDataset``

.. code:: python

   from mowl.datasets.builtin import PPIYeastSlimDataset

   dataset = PPIYeastSlimDataset()


For projecting the ontology, mOWL provides different :doc:`ontology projection methods <../api/projection/index>`. In this case let's use DL2Vec projection method, which can be imported and defined in two different ways:

The first way is calling the ``DL2VecProjector`` class directly:

.. code:: python

   from mowl.projection.dl2vec.model import DL2VecProjector
   projector = DL2VecProjector(bidirectional_taxonomy = True)
   edges = projector.project(dataset.ontology)

The second way is calling the class through a factory method:

.. code:: python
	  
   from mowl.projection.factory import projector_factory
   projector = projector_factory("dl2vec", bidirectional_taxonomy = True)
   edges = projector.project(dataset.ontology)

The factory method will return a different projector class depending on the first parameter.

After obtaining a graph as an edge list, we can perform random walks. As before, the walking methods can be imported directly or through a factory method. Let's use the factory method in this case.
   
.. code:: python
   
   from mowl.walking.factory import walking_factory
   walker = walking_factory("deepwalk", alpha = 0.1, walk_length = 10, num_walks = 10, outfile = "data/walks")
   walker.walk(edges)

The walks are saved in the specified filepath. After generating the walks, one usual step is to use the Word2Vec model to generate embeddings of the entities. In this case we rely on the library Gensim to use the Word2Vec model.
   
.. code:: python
   
   from gensim.models import Word2Vec
   from gensim.models.word2vec import LineSentence
   

   corpus = LineSentence(walker.outfile)
   
   w2v_model = Word2Vec(
	  corpus,
	  sg=1,
          min_count=1,
          vector_size=10,
          window = 10,
          epochs = 10,
          workers = 16)

   vectors = w2v_model.wv


Translational embeddings
--------------------------

Translational embeddings were designed to embed knowedge graphs. Since we can generate a graph representation of an ontology we can also perform translational embeddings.

To provide translational embedding methods in mOWL, we rely on `PyKEEN <https://github.com/pykeen/pykeen>`_, which contains efficient implementations of those methods.


After generating a set of `edges`, we can call a translational method in the following way:

.. code:: python

   from mowl.embeddings.translational.model import TranslationalOnt

   trans_model = TranslationalOnt(
        edges,
        trans_method = "transE",
        embedding_dim = 50,
        epochs = 20,
        batch_size = 32
    )

    trans_model.train()
    embeddings = transMethod.get_embeddings()

.. note::

   Notice the parameter ``trans_method`` is set to ``transE``. There are other 3 options: ``transH``, ``transR``, ``transD``, each of which corresponds to different variations of translational methods.

   
 
   
Syntactic embeddings
------------------------

This approach consists on generating textual representations (sentences) from ontologies. For this task, we provide methods like ``extract_axiom_corpus`` that generates sentences out of axioms in the ontology. Furthermore, the method ``extract_annotation_corpus`` will generate corpus from the annotations in the ontology.

As a data augmentation step, reasoning can be applied to the ontology to generate more axioms. For reasoning, the methods of the OWLAPI can be accesed directly. However, we provide the wrapper class ``MOWLReasoner`` with implementation of some common tasks, such as inferring subclass, equivalent class and disjoint class axioms.

The following code example corresponds to the implementation of the paper `Onto2Vec: joint vector-based representation of biological entities and their ontology-based annotations <https://academic.oup.com/bioinformatics/article/34/13/i52/5045776>`_

First, we need to do the corresponding imports

.. code:: python

   from mowl.reasoning.base import MOWLReasoner
   from org.semanticweb.elk.owlapi import ElkReasonerFactory

Then, we perform the reasoning steps to add axioms to the training ontology.

.. code:: python
	  
   reasoner_factory = ElkReasonerFactory()
   reasoner = reasoner_factory.createReasoner(dataset.ontology)
   reasoner.precomputeInferences()

   mowl_reasoner = MOWLReasoner(reasoner)
   mowl_reasoner.infer_subclass_axioms(dataset.ontology)
   mowl_reasoner.infer_equiv_class_axioms(dataset.ontology)

After preprocessing the ontology, we generate the corpus out of the ontology axioms and save the corpus into a file.

.. code:: python

   from mowl.corpus.base import extract_and_save_axiom_corpus
   extract_and_save_axiom_corpus(dataset.ontology, "corpus_file_path")

      
Finally, use Word2vec to generate the embeddings

.. code:: python
   
   sentences = LineSentence(corpus_file)

   model = Word2Vec(
            sentences,
            sg = 1,
            min_count = 1,
            vector_size = 20,
            window = 5,
            epochs = 20,
            workers = 4,
	    device = "cuda"
        )

   model.save(word2vec_file)


To implement the paper `OPA2Vec: combining formal and informal content of biomedical ontologies to improve similarity-based prediction <https://pubmed.ncbi.nlm.nih.gov/30407490/>`_, we replace the code

.. code:: python
	  
   extract_and_save_axiom_corpus(dataset.ontology, "corpus_file_path")

with

.. code:: python

   from mowl.corpus.base import extract_and_save_axiom_corpus, extract_and_save_annotation_corpus

   extract_and_save_axiom_corpus(dataset.ontology, "corpus_file_path")
   extract_and_save_annotation_corpus(ds.ontology, "corpus_file_path", mode = "a")



To add annotation textual information to the corpus. The parameter `mode` will be used for opening the file where the axioms will be saved. In the example above, we want the annotations axioms to be added to the file where the class axioms where already written.




Semantic embedddings
----------------------

This type of embeddings are generated with models designed to capture the semantics of the ontology axioms.


One example of this models is `EL Embeddings <https://www.ijcai.org/Proceedings/2019/845>`_. To use this method we can write the following code:

.. code:: python

   from mowl.embeddings.elembeddings.model import ELEmbeddings

   model = ELEmbeddings(
        dataset,
        epochs = 1000,
        margin = 0.1,
        model_filepath = "model.th",
	device = "cuda"
    )

   model.train()

   embeddings = model.get_embeddings()
