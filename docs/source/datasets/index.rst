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

To access any of these datasets you can use:

.. testcode::

   from mowl.datasets.builtin import PPIYeastSlimDataset
   ds = PPIYeastSlimDataset()
   train_ontology = ds.ontology
   valid_ontology = ds.validation
   test_ontology = ds.testing

   evaluation_classes = ds.evaluation_classes

Built-in datasets already contain the attribute `evaluation_classes`, which is used to evaluate a model on the dataset. In the PPI example, the evaluation classes correspong to ontology classes representing proteins.


   
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
          
