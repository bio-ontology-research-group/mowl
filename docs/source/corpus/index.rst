Ontology to text
==================

In this section we show how mOWL can be used to generate a text corpus out of an ontology. We provide two methods, one for rendering the logical axioms and another for rendering annotation axioms. Both approaches generate a **Manchester Syntax** representations of the axioms.

Rendering logical axioms
--------------------------

To generate a corpus out of the logical axioms, we can use the following snippet:

.. testcode::

   from mowl.corpus import extract_axiom_corpus
   from mowl.datasets.builtin import FamilyDataset

   dataset = FamilyDataset()
   corpus = extract_axiom_corpus(dataset.ontology)

In this way we will get a list of strings where each one will be a Manchester Syntax rendering of an axiom. In case the use case is to save the corpus to disk, the next line could be used:

.. testcode::

   from mowl.corpus import extract_and_save_axiom_corpus
   extract_and_save_axiom_corpus(dataset.ontology,
                                 "/tmp/file_to_save_corpus",
				 mode="w")

.. hint::

   Parameter ``mode`` reflects how to write to the file. ``mode="w"`` would overwrite the file and ``mode="a"`` would append to the existing contents of the file.


Rendering annotations from ontology
-------------------------------------

Annotations from ontology can be also rendered in a similar way. To extract the annotations, use the following example:

.. testcode::

   from mowl.datasets.builtin import PPIYeastSlimDataset
   from mowl.corpus import extract_annotation_corpus

   dataset = PPIYeastSlimDataset()
   corpus = extract_annotation_corpus(dataset.ontology)

And to save into a file:

.. testcode::

   from mowl.corpus import extract_and_save_annotation_corpus
   extract_and_save_annotation_corpus(dataset.ontology,
                                 "/tmp/file_to_save_corpus",
				 mode="w")




Embedding ontologies
----------------------

.. note::
   This feature has been added since version 0.2.0

To train a Word2Vec model with a generated corpus, we can use the class: :class:`SyntacticPlusW2VModel <mowl.models.syntactic.w2v_model.SyntacticPlusW2VModel>`:

.. testcode::

   from mowl.models import SyntacticPlusW2VModel
   model = SyntacticPlusW2VModel(dataset, corpus_filepath="test")
   model.set_w2v_model(min_count=1)
   model.generate_corpus(save=True, with_annotations=True)
   model.train()

.. testoutput::

   Corpus saved in test
 
.. attention::
   The ``set_w2v_model`` receives the same arguments as the :class:`gensim.models.word2vec.Word2Vec` model.

