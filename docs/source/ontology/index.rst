Ontology management
=========================
.. testsetup:: 

   from org.semanticweb.owlapi.model import IRI
   from mowl.owlapi import OWLAPIAdapter
   manager = OWLAPIAdapter().owl_manager
   ontology = manager.createOntology()
   manager.saveOntology(ontology, IRI.create("file:" + os.path.abspath("MyOntology.owl")))

   with open("annots.tsv", "w") as f:
       f.write("hi")

   with open("my_triples_file.tsv", "w") as f:
       f.write("node1\trel1\tnode2")

   
Adding axioms to ontologies
----------------------------------

The method :func:`insert_annotations <mowl.ontology.extend.insert_annotations>` allows adding new axioms in the form :math:`C \sqsubseteq \exists R.D`. Entities :math:`C`, :math:`R` and :math:`D` must be stored in a ``.tsv`` file.

For example, let's say we have an ontology called :download:`MyOntology.owl` and we want to add new axioms with a relation ``http://has_annotation``.

The axiom information must be stored in an :download:`annotations file <annots.tsv>` with the following format:

.. code:: text

   http://prefix1/class1    	http://prefix3/class3
   http://prefix2/class2	    http://prefix4/class4	    http://prefix5/class5
   
Then to add that information to the ontology we use the following instructions:
   
.. testcode:: 

   from mowl.ontology.extend import insert_annotations
   annotation_data_1 = ("annots.tsv", "http://has_annotation", True)
   annotations = [annotation_data_1] # There  could be more than 1 annotations file.
   insert_annotations("MyOntology.owl", annotations, out_file = None)

The annotations will be added to the ontology and since ``out_file = None``, the input ontology will be overwritten.

.. note::
   Notice that the variable ``annotation_document_1`` has three elements. The first is the path of the annotations document, the second is the label of the relation for all the annotations and the third is a parameter indicating if the annotation is directed or not; in the case it is set to ``False``, the axiom will be added in both *directions* (:math:`C \sqsubseteq \exists R.D` and :math:`D \sqsubseteq \exists R.C`).

In our example, the axioms inserted in the ontology will be the following in XML/OWL format:

.. code:: xml

	  <Class rdf:about="http://prefix1/class1">
        <rdfs:subClassOf>
            <Restriction> 
                <onProperty rdf:resource="http:///has_annotation"/>
		<someValuesFrom rdf:resource="http://prefix2/class2"/>
            </Restriction>
        </rdfs:subClassOf>
    </Class>

   <Class rdf:about="http:///prefix2/class2">
        <rdfs:subClassOf>
            <Restriction>
                <onProperty rdf:resource="http:///has_annotation"/>
                <someValuesFrom rdf:resource="http://prefix4/class4"/>
            </Restriction>
        </rdfs:subClassOf>
    </Class>

   <Class rdf:about="http:///prefix2/class2">
        <rdfs:subClassOf>
            <Restriction>
                <onProperty rdf:resource="http:///has_annotation"/>
                <someValuesFrom rdf:resource="http://prefix5/class5"/>
            </Restriction>
        </rdfs:subClassOf>
    </Class>


Creating ontology from triples
-----------------------------------------------

To transform a triples from a ``.tsv`` file into a ``.owl``, we can do using the :func:`create_from_triples <mowl.ontology.create.create_from_triples>` method. As before, an input triple ``(h,r,t)`` will be inserted as axioms of the form :math:`H \sqsubseteq \exists R.T`.

Let's assume we have a triples file called :download:`my_triples_file.tsv`  of the following form:

.. code:: text

   http://mowl/class1    http://mowl/relation1    http://mowl/class2
   http://mowl/class2    http://mowl/relation4    http://mowl/class3
   http://mowl/class5    http://mowl/relation2    http://mowl/class2
   http://mowl/class1    http://mowl/relation1    http://mowl/class3

To create an ontology from those triples we would do:

.. testcode::

   from mowl.ontology.create import create_from_triples

   triples_file = "my_triples_file.tsv"
   out_file = "my_new_ontology.owl"

   create_from_triples(triples_file, out_file)


In case we have a :download:`simpler triples file <simpler_triples_file.tsv>` like the following:

.. code:: text

   class1    class2
   class2    class3
   class5    class2
   class1    class3

we can create an ontology assuming all the triples will have the same relation and also inputting a prefix for all the classes:

.. testcode::

   from mowl.ontology.create import create_from_triples

   triples_file = "simpler_triples_file.tsv"
   out_file = "my_new_ontology.owl"
   prefix = "http://mowl/"
   relation = "http://mowl/relation"

   create_from_triples(triples_file,
                       out_file,
		       relation_name = relation,
		       bidirectional = True,
		       head_prefix=prefix,
		       tail_prefix=prefix)


.. note::

   The ``bidirectional`` parameter indicates whether the graph will be directed or undirected.


:math:`\mathcal{EL}` normalization
--------------------------------------

The :math:`\mathcal{EL}` language is part of the Description Logics family. Concept descriptions in :math:`\mathcal{EL}` can be expressed in the following normal forms:

.. math::
   \begin{align}
   C &\sqsubseteq D & (\text{GCI 0}) \\
   C_1 \sqcap C_2 &\sqsubseteq D & (\text{GCI 1}) \\
   C &\sqsubseteq \exists R. D & (\text{GCI 2})\\
   \exists R. C &\sqsubseteq D & (\text{GCI 3}) 
   \end{align}

   
.. hint::

   GCI stands for General Concept Inclusion

The bottom concept can exist in the right side of GCIs 0,1,3 only, which can be considered as special cases and extend the normal forms to include the following:

.. math::
   \begin{align}
   C &\sqsubseteq \bot & (\text{GCI BOT 0}) \\
   C_1 \sqcap C_2 &\sqsubseteq \bot & (\text{GCI BOT 1}) \\
   \exists R. C &\sqsubseteq \bot & (\text{GCI BOT 3}) 
   \end{align}


We rely on `JCEL <https://julianmendez.github.io/jcel/>`_ to provide :math:`\mathcal{EL}` normalization by wrapping into the mOWL's :class:`ELNormalizer <mowl.ontology.normalize.ELNormalizer>`

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.ontology.normalize import ELNormalizer, GCI

   ontology = FamilyDataset().ontology
   normalizer = ELNormalizer()
   gcis = normalizer.normalize(ontology)


The resulting variable ``gcis`` is a dictionary of the form:

+------------+--------------------------------------------------------------+
| Key        | Value                                                        |
+============+==============================================================+
| "gci0"     | list of :class:`GCI0 <mowl.ontology.normalize.GCI0>`         |
+------------+--------------------------------------------------------------+
| "gci1"     | list of :class:`GCI1 <mowl.ontology.normalize.GCI1>`         |
+------------+--------------------------------------------------------------+
| "gci2"     | list of :class:`GCI2 <mowl.ontology.normalize.GCI2>`         |
+------------+--------------------------------------------------------------+
| "gci3"     | list of :class:`GCI3 <mowl.ontology.normalize.GCI3>`         |
+------------+--------------------------------------------------------------+
| "gci0_bot" | list of :class:`GCI0_BOT <mowl.ontology.normalize.GCI0_BOT>` |
+------------+--------------------------------------------------------------+
| "gci1_bot" | list of :class:`GCI1_BOT <mowl.ontology.normalize.GCI1_BOT>` |
+------------+--------------------------------------------------------------+
| "gci3_bot" | list of :class:`GCI3_BOT <mowl.ontology.normalize.GCI3_BOT>` |
+------------+--------------------------------------------------------------+


