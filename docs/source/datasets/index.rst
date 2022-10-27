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



mOWL is designed to handle input in OWL format. That is, you can input OWL ontologies. A mOWL dataset contains 3 ontologies: training, validation, testing.

Built-in datasets
-------------------

There are several built-in datasets related to bioinformatics tasks such as protein-protein interactions prediction and gene-disease association prediction. This datasets are:

* :class:`FamilyDataset <mowl.datasets.builtin.FamilyDataset>`
* :class:`PPIYeastDataset <mowl.datasets.builtin.PPIYeastDataset>`
* :class:`PPIYeastSlimDataset <mowl.datasets.builtin.PPIYeastSlimDataset>`
* :class:`GDAHumanDataset <mowl.datasets.builtin.GDAHumanDataset>`
* :class:`GDAMouseDataset <mowl.datasets.builtin.GDAMouseDataset>`
* :class:`GDAHumanELDataset <mowl.datasets.builtin.GDAHumanELDataset>`
* :class:`GDAMouseELDataset <mowl.datasets.builtin.GDAMouseELDataset>`

To access any of these datasets you can use:

.. testcode::

   from mowl.datasets.builtin import PPIYeastSlimDataset
   ds = PPIYeastSlimDataset()
   train_ontology = ds.ontology
   valid_ontology = ds.validation
   test_ontology = ds.testing

   
Your own dataset
--------------------------

In case you have your own training, validation and testing ontologies, you can turn them easily to a mOWL dataset as follows:

.. testcode::

   from mowl.datasets.base import PathDataset
   ds = PathDataset("training_ontology.owl", 
                    validation_path="validation_ontology.owl", testing_path="testing_ontology.owl")
   

.. note::
   Validation and testing ontologies are optional when using :class:`PathDataset <mowl.datasets.base.PathDataset>`. By default they are set to ``None``.
   
