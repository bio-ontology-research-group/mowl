Datasets
==========

.. testsetup::

   from org.semanticweb.owlapi.model import IRI
   from mowl.owlapi import OWLAPIAdapter
   manager = OWLAPIAdapter().owl_manager
   train_ont = manager.createOntology()
   valid_ont = manager.createOntology()
   test_ont = manager.createOntology()

   manager.saveOntology(train_ont, IRI.create("file:" + os.path.abspath("training_ontology.owl")))
   manager.saveOntology(valid_ont, IRI.create("file:" + os.path.abspath("validation_ontology.owl")))
   manager.saveOntology(test_ont, IRI.create("file:" + os.path.abspath("testing_ontology.owl")))



mOWL is designed to handle input in OWL format. That is, you can input OWL ontologies. A mOWL dataset contains 3 ontologies: training, validation and testing.

Built-in datasets
-------------------

There are several built-in datasets related to bioinformatics tasks such as protein-protein interactions prediction and gene-disease association prediction. Datasets can be found at :doc:`Datasets API docs <../../api/datasets/index>`.

For :math:`\mathcal{EL}^{++}` subsumption prediction, mOWL provides the GALEN, GO and Anatomy ontologies with the 80/10/10 benchmark split distributed with the reference implementation of [jackermeier2024]_, also used by the TransBox paper (WWW 2025):

.. testcode::

   from mowl.datasets.builtin import GALENJackermeier2024Dataset
   ds = GALENJackermeier2024Dataset()
   train_ontology = ds.ontology
   valid_ontology = ds.validation
   test_ontology = ds.testing

The training ontologies also contain :math:`\mathcal{EL}^{++}` role inclusion and role chain axioms. Models built on top of :class:`EmbeddingELModel <mowl.base_models.EmbeddingELModel>` train on them when constructed with ``load_role_axioms=True`` and their module implements the two role axiom losses (see :doc:`the EL guide </embedding_el/index>`).

The axioms and the split are the frozen release of the reference benchmark, which distributes
them as integer-indexed splits plus the class and relation index files; mOWL serves the same
axioms serialized as OWL. The terms of the underlying ontologies and of the reference
benchmark apply to this redistribution.

.. warning::
   In ``AnatomyJackermeier2024Dataset`` the class IRIs are **not** anatomy identifiers: the
   reference benchmark ships no class names for ANATOMY, so every class is an opaque
   generated IRI under ``http://bio2vec.net/data/mowl/anatomy_jackermeier2024/aux/``. The
   axioms and the split are faithful, so ranking metrics are comparable, but predictions
   cannot be mapped back to anatomical terms. GALEN and GO keep their original IRIs.


To access any of these datasets you can use:

.. testcode::

   from mowl.datasets.builtin import PPIYeastSlimDataset
   ds = PPIYeastSlimDataset()
   train_ontology = ds.ontology
   valid_ontology = ds.validation
   test_ontology = ds.testing

   evaluation_classes = ds.evaluation_classes

Built-in datasets already contain the attribute `evaluation_classes`, which is used to evaluate a model on the dataset. In the PPI example, the evaluation classes correspong to ontology classes representing proteins.


Where datasets are stored
---------------------------

.. versionchanged:: 2.2.0
   Built-in datasets used to be downloaded into the current working directory. They are now cached in a user-level directory, shared by every working directory.

A built-in dataset is downloaded once and reused afterwards. The download location is resolved by :func:`default_data_root <mowl.datasets.default_data_root>`:

.. testcode::

   from mowl.datasets import default_data_root
   cache_dir = default_data_root()

This is ``$XDG_CACHE_HOME/mowl/datasets``, falling back to ``~/.cache/mowl/datasets``. Set the ``MOWL_DATA_ROOT`` environment variable to override it globally:

.. code-block:: bash

   export MOWL_DATA_ROOT=/data/shared/mowl

Or pass ``data_root`` to a single dataset:

.. testcode::

   import tempfile
   from mowl.datasets.builtin import FamilyDataset
   ds = FamilyDataset(data_root=tempfile.mkdtemp())


Your own dataset
--------------------------

In case you have your own :download:`training <training_ontology.owl>`, :download:`validation <validation_ontology.owl>` and :download:`testing <testing_ontology.owl>` ontologies, you can turn them easily to a mOWL dataset as follows:

.. testcode::

   from mowl.datasets.base import PathDataset
   ds = PathDataset("training_ontology.owl", 
                    validation_path="validation_ontology.owl",
		    testing_path="testing_ontology.owl")

   training_axioms = ds.ontology.getAxioms()
   validation_axiom = ds.validation.getAxioms()
   testing_axioms = ds.testing.getAxioms()

.. note::
   Validation and testing ontologies are optional when using :class:`PathDataset <mowl.datasets.base.PathDataset>`. By default they are set to ``None``.
   
.. attention::

   Custom datasets require the implementation of the `evaluation_classes` attribute. This can be done as:

.. code:: python

   class CustomDataset(PathDataset):
       def __init__(self, *args, **kwargs):
           super().__init__(train_path, valid_path, test_path)

      @property
      def evaluation_classes(self):
          #################
	  # your code here
	  #################
          
