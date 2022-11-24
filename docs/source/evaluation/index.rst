Evaluating the embeddings
===============================

.. |modbasedevaluator| replace:: :class:`ModelRankBasedEvaluator <mowl.evaluation.rank_based.ModelRankBasedEvaluator>`
.. |embbasedevaluator| replace:: :class:`EmbeddingsBasedEvaluator <mowl.evaluation.rank_based.EmbeddingsRankBasedEvaluator>`
				

After training our models, evaluation would be the next step. This is the first version of the evaluation module. We provide the class :class:`RankBasedEvaluator <mowl.evaluation.rank_based.RankBasedEvaluator>` that is divided into two types of evaluators:

* |embbasedevaluator|
* |modbasedevaluator|

Embeddings-based evaluator
-----------------------------

This class is intended to be used by models or workflows in which the embedding vectors exist outside some model class. For example, see :doc:`/examples/graph_based/plot_1_dl2vec`. In that example, the embeddings are not enclosed in any model and therefore, some preprocessing is needed to determine the information that the |embbasedevaluator| requires.

Let's use this evaluator with a simpler version of :doc:`Onto2Vec </examples/syntactic/plot_1_onto2vec>` with the protein-protein interaction task:

.. testcode:: eval1

   import mowl
   mowl.init_jvm("10g")

   from mowl.datasets.builtin import PPIYeastSlimDataset
   from mowl.corpus import extract_and_save_axiom_corpus
   from mowl.projection import TaxonomyWithRelationsProjector
   from gensim.models.word2vec import LineSentence
   from gensim.models import Word2Vec
   
   dataset = PPIYeastSlimDataset()
   extract_and_save_axiom_corpus(dataset.ontology, "onto2vec_corpus")

   corpus = LineSentence("onto2vec_corpus")
   w2v_model = Word2Vec(corpus, epochs=5, vector_size=10)


.. testcode:: eval1
   
   from mowl.evaluation.rank_based import EmbeddingsRankBasedEvaluator
   from mowl.evaluation.base import CosineSimilarity
   from mowl.projection import TaxonomyWithRelationsProjector

   proteins = dataset.evaluation_classes

   projector = TaxonomyWithRelationsProjector(taxonomy=False,
                                              relations=["http://interacts_with"])

   evaluation_edges = projector.project(dataset.testing)
   filtering_edges = projector.project(dataset.ontology)


The gene-disease associations will be scoredc using cosine similarity. For that reason we use the ``CosineSimilarity`` class.

.. testcode:: eval1

   vectors = w2v_model.wv
   evaluator = EmbeddingsRankBasedEvaluator(vectors,
                                            evaluation_edges,
					    CosineSimilarity,
					    training_set=filtering_edges,
					    head_entities = proteins.as_str,
					    tail_entities = proteins.as_str,
					    device = 'cpu'
					    )

.. code:: python
	  
   evaluator.evaluate(show=True)


The output will look like the following:

.. code:: bash

   Hits@1:   0.00   Filtered:   0.01
   Hits@10:  0.02   Filtered:   0.10
   Hits@100: 0.20   Filtered:   0.33
   MR:       933.38 Filtered: 877.85
   AUC:      0.84   Filtered:   0.85




Model-based evaluator
------------------------

This class is intended to be used by models or workflows where (a) the embedding vectors are enclosed in an embeddings model, (b) the embedding information of a particular entity is encoded in more than one embedding vector. For instance, check :doc:`/examples/elmodels/plot_2_elboxembeddings`, where the embedding for a class :math:`C` consists on two things: one embedding vector for the center of box and another embedding vector for the offset components. In this case, |modbasedevaluator| is more suitable since in the end, we are just interested on inputting the class names or indicex into the evaluator and do not care about how the information about it is handled.


For a detailed example of the use of this evaluator please refer to :doc:`/examples/elmodels/plot_2_elboxembeddings`

.. warning::

   This is the first version of the module for evaluating embeddings. This module will be deprecated soon as a new module is on development process.
