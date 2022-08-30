Datasets
==========

mOWL is designed to handle input in OWL format. That is, you can input OWL ontologies. A mOWL dataset contains 3 ontologies: training, validation, testing.

Built-in datasets
-------------------

There are several built-in datasets related to bioinformatics tasks such as protein-protein interactions prediction and gene-disease association prediction. This datasets are:

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
   ds = PathDataset("training_ontology.owl", "validation_ontology.owl", "testing_ontology.owl")
   

.. note::
   Validation and testing ontologies are optional when using :class:`PathDataset <mowl.datasets.base.PathDataset>`. By default they are set to ``None``.
   
Adding annotations to ontologies
----------------------------------

To add annotations in the form of axioms there is the method ``insert_annotations``. All the annotations will be inserted into the ontology in the form :math:`C \sqsubseteq \exists R.D`, where :math:`C` is the annotating entity (it can be a new ontology class), :math:`D` is the annotated entity (usually is a class already existing in the ontology) and :math:`R` is the label of the relation. The annotation information must be stored in a ``.tsv`` file.

For example, let's say we have an ontology called ``MyOntology.owl`` where there are ontology classes ``http://some_prefix/class_001``, ``http://some_prefix/class_002`` and ``http://some_prefix/class_003``. Furthermore, we have some other classes ``http://newclass1``, ``http://newclass2`` that are in relation with the already classes in then ontology. The relation must have a label (let's use ``http://has_annotation``).

The annotation information must be store in a file with the following format:

.. code:: text

   http://newclass1    http://some_prefix/class:002
   http://newclass2    http://some_prefix/class:001    http://some_prefix/class:003

Then to add that information to the ontology we use the following instructions:
   
.. code:: python

   from mowl.ontology.extend import insert_annotations
   anotation_data_1 = ("annots.tsv", "http://has_annotation", True)
   annotations = [annotation_document_1] # There  could be more than 1 annotations file.
   insert_annotations("MyOntology.owl", annotations, out_file = None)

The annotations will be added to the ontology and since ``out_file = None``, the input ontology will be overwritten.

.. note::
   Notice that the variable ``annotation_document_1`` has three elements. The first is the path of the annotations document, the second is the label of the relation for all the annotations and the third is a parameter indicating if the annotation is directed or not; in the case it is set to ``False``, the axiom will be added in both *directions* (:math:`C \sqsubseteq \exists R.D` and :math:`D \sqsubseteq \exists R.C`).

In our example, the axioms inserted in the ontology have the form:

.. code:: xml

   <Class rdf:about="http://newclass1">
        <rdfs:subClassOf>
            <Restriction> 
                <onProperty rdf:resource="http:///has_annotation"/>
                <someValuesFrom rdf:resource="http://some_prefix/class:002"/>
            </Restriction>
        </rdfs:subClassOf>
    </Class>

   <Class rdf:about="http:///newclass2">
        <rdfs:subClassOf>
            <Restriction>
                <onProperty rdf:resource="http:///has_annotation"/>
                <someValuesFrom rdf:resource="http://some_prefix/class:001"/>
            </Restriction>
        </rdfs:subClassOf>
    </Class>

   <Class rdf:about="http:///newclass2">
        <rdfs:subClassOf>
            <Restriction>
                <onProperty rdf:resource="http:///has_annotation"/>
                <someValuesFrom rdf:resource="http://some_prefix/class:003"/>
            </Restriction>
        </rdfs:subClassOf>
    </Class>







   



