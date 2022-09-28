Datasets
==========

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

.. code-block:: python

   from mowl.datasets.builtin import PPIYeastDataset

   ds = PPIYeastDataset()
   train_ontology = ds.ontology
   valid_ontology = ds.validation
   test_ontology = ds.testing

Your own dataset
--------------------------

In case you have your own training, validation and testing ontologies, you can turn them easily to a mOWL dataset as follows:

.. code-block:: python

   from mowl.datasets.base import PathDataset
   ds = PathDataset("training_ontology.owl", 
                    validation_path="validation_ontology.owl", testing_path="testing_ontology.owl")
   

.. note::
   Validation and testing ontologies are optional when using :class:`PathDataset <mowl.datasets.base.PathDataset>`. By default they are set to ``None``.
   
