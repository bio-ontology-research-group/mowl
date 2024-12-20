
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "examples/syntactic/plot_2_opa2vec.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_examples_syntactic_plot_2_opa2vec.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_syntactic_plot_2_opa2vec.py:


OPA2Vec
===========

This example corresponds to the paper `OPA2Vec: combining formal and informal content of biomedical ontologies to improve similarity-based prediction <https://doi.org/10.1093/bioinformatics/bty933>`_. 

This method is an extension of Onto2Vec that apart from formal knowldege (i.e. axioms) it also uses informal knowledge such as entity metadata (i.e. synonyms, definitions, etc.)

.. GENERATED FROM PYTHON SOURCE LINES 14-20

For this algorithm, we need four components:

- The reasoner
- The corpus generator
- The annotations generator
- The Word2Vec model

.. GENERATED FROM PYTHON SOURCE LINES 20-38

.. code-block:: Python



    import mowl
    mowl.init_jvm("20g")

    from mowl.datasets.builtin import PPIYeastSlimDataset
    from mowl.corpus import extract_and_save_axiom_corpus, extract_and_save_annotation_corpus
    from mowl.owlapi import OWLAPIAdapter
    from mowl.reasoning import MOWLReasoner

    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    from java.util import HashSet

    from gensim.models.word2vec import LineSentence
    from gensim.models import Word2Vec

    import os








.. GENERATED FROM PYTHON SOURCE LINES 39-44

Inferring new axioms
--------------------

OPA2Vec uses an ontology reasoner to infer new axioms as a preprocessing step. In the original
paper, the authors used the HermiT reasoner. For this example, we use the ELK reasoner.

.. GENERATED FROM PYTHON SOURCE LINES 44-51

.. code-block:: Python


    dataset = PPIYeastSlimDataset()

    reasoner_factory = ElkReasonerFactory()
    reasoner = reasoner_factory.createReasoner(dataset.ontology)
    mowl_reasoner = MOWLReasoner(reasoner)








.. GENERATED FROM PYTHON SOURCE LINES 52-59

We wrap the reasoner into the :class:`MOWLReasoner <mowl.reasoning.base.MOWLReasoner>` class \
in order to use some shortcuts the mOWL
provides such as:

- inferring subclass axioms
- inferring equivalent class axioms
- inferring disjoint axioms (not applicable for this example since we use ELK reasoner)

.. GENERATED FROM PYTHON SOURCE LINES 59-64

.. code-block:: Python


    classes = dataset.ontology.getClassesInSignature()
    subclass_axioms = mowl_reasoner.infer_subclass_axioms(classes)
    equivalent_class_axioms = mowl_reasoner.infer_equivalent_class_axioms(classes)








.. GENERATED FROM PYTHON SOURCE LINES 65-66

We can now add the inferred axioms to the ontology.

.. GENERATED FROM PYTHON SOURCE LINES 66-77

.. code-block:: Python


    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager

    axioms = HashSet()
    axioms.addAll(subclass_axioms)
    axioms.addAll(equivalent_class_axioms)

    manager.addAxioms(dataset.ontology, axioms)






.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    <java object 'org.semanticweb.owlapi.model.parameters.ChangeApplied'>



.. GENERATED FROM PYTHON SOURCE LINES 78-83

Generating the corpus and training the model
-----------------------------------------------

Now that we have an extended ontology, we can generate the corpus out of it. After that, we
can train the Word2Vec model.

.. GENERATED FROM PYTHON SOURCE LINES 83-90

.. code-block:: Python


    extract_and_save_axiom_corpus(dataset.ontology, "opa2vec_corpus.txt")
    extract_and_save_annotation_corpus(dataset.ontology, "opa2vec_corpus.txt", mode="a")

    sentences = LineSentence("opa2vec_corpus.txt")
    model = Word2Vec(sentences, vector_size=5, window=2, min_count=1, workers=4, epochs=2)








.. GENERATED FROM PYTHON SOURCE LINES 91-92

Cleaning up the memory

.. GENERATED FROM PYTHON SOURCE LINES 92-94

.. code-block:: Python


    os.remove("opa2vec_corpus.txt")








.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 14.199 seconds)

**Estimated memory usage:**  405 MB


.. _sphx_glr_download_examples_syntactic_plot_2_opa2vec.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_2_opa2vec.ipynb <plot_2_opa2vec.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_2_opa2vec.py <plot_2_opa2vec.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_2_opa2vec.zip <plot_2_opa2vec.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
