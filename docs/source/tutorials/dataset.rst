Creating datasets
====================

In mOWL, the :doc:`datasets <../api/datasets/index>` are objects with three attributes: ``ontology``, ``validation`` and ``testing``. The three attributes are OWL ontologies and correspond to training, validation and testing information, respectively.

There are different methods to declare a ``Dataset`` object. An example is using the ``PathDataset`` class, for which the ontologies must be stored locally as ``.owl`` files.

.. code:: python

   from mowl.datasets.base import PathDataset

   ds = PathDataset("training_ontology.owl", "validation_ontology.owl", "testing_ontology.owl")


.. note::

   The validation and testing ontologies are optional and can be set to ``None``.


Adding annotations to ontologies
----------------------------------

To add annotations in the form of axioms there is the method ``insert_annotations``. All the annotations will be inserted into the ontology in the form :math:`C \sqsubseteq \exists R.D`, where :math:`C` is the annotating entity (it can be a new ontology class), :math:`D` is the annotated entity (usually is a class already existing in the ontology) and :math:`R` is the label of the relation annotation. The annotation information must be stored in a ``tsv`` file.

For example, let's say we have an ontology called ``MyOntology.owl`` where there are ontology classes ``http://some_prefix/class_001``, ``http://some_prefix/class_002`` and ``http://some_prefix/class_003``. Furthermore, we have some other classes ``http://newclass1``, ``http://newclass2`` that are in relation with the already classes in then ontology. The relation must have a label (let's use ``http://has_annotation``).

The annotation information must be store in a file with the following format:

.. code:: text

   http://newclass1    http://some_prefix/class:002
   http://newclass2    http://some_prefix/class:001    http://some_prefix/class:003

Then to add that information to the ontology we use the following instructions:

.. code:: python

   anotation_document_1 = ("annots.tsv", "http://has_annotation", True)
   annotations = [annotation_document_1]
   insert_annotations("MyOntology.owl", annotations, out_file = None)

The annotations will be added to the ontology and since ``out_file = None``, the input ontology will be overwritten.

.. note::
   Notice that the variable ``annotation_document_1`` has three elements. The first is the path of the annotations document, the second is the label of the relation for all the annotations and the third is a parameter indicating if the annotation is directed or not; in the case it is set to False, the axiom will be added in both directions (:math:`C \sqsubseteq \exists R.D` and :math:`D \sqsubseteq \exists R.C`).

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







   



