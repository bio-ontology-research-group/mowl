OWLAPI
=========

.. testsetup::

   import os
   from mowl.owlapi import OWLAPIAdapter
   from org.semanticweb.owlapi.model import IRI
   manager = OWLAPIAdapter().owl_manager
   ontology = manager.createOntology()
   manager.saveOntology(ontology, IRI.create("file:" + os.path.abspath("my_ontology.owl")))
 


mOWL interfaces the OWLAPI by using JPype to connect to the Java Virtual Machine. JPype allows to access Java objects and methods from a Python script. For example, we can use the OWLAPI from Python:

.. testcode::
   
   import mowl
   mowl.init_jvm("10g")
   from org.semanticweb.owlapi.apibinding import OWLManager
   from java.io import File

   input_file = File("my_ontology.owl")
   manager = OWLManager.createOWLOntologyManager()
   ontology = manager.loadOntologyFromOntologyDocument(input_file)

   ontology.getClassesInSignature()

.. note::

   Notice that accessing to Java objects is only possible after starting the JVM


Shortcuts to the OWLAPI
--------------------------------

In order to ease the ontology management, mOWL provides an adapter class that wraps some objects found in the OWLAPI such as the OWLManager and the DataFactory:

.. testcode::

   from mowl.owlapi import OWLAPIAdapter

   adapter = OWLAPIAdapter()
   owl_manager = adapter.owl_manager # Instance of org.semanticweb.owlapi.apibinding.OWLManager
   data_factory = adapter.data_factory # Equivalent to owl_manager.getOWLDataFactory()



Shortcuts to OWL Reasoners
------------------------------

mOWL provides some shortcuts for performing reasoning over ontologies. The works by wrapping OWL Reasoners that are instances of ``org.semanticweb.owlapi.reasoner.OWLReasoner``. The following example shows how to obtain inferences from an ontology.

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.reasoning.base import MOWLReasoner
   from org.semanticweb.HermiT import Reasoner

   dataset = FamilyDataset()

   reasoner = Reasoner.ReasonerFactory().createReasoner(dataset.ontology)
   reasoner.precomputeInferences()
 
   mowl_reasoner = MOWLReasoner(reasoner)
   classes_to_infer_over = list(dataset.ontology.getClassesInSignature())
  
   subclass_axioms = mowl_reasoner.infer_subclass_axioms(classes_to_infer_over)
   equivalence_axioms = mowl_reasoner.infer_equivalent_class_axioms(classes_to_infer_over)
   disjointness_axioms = mowl_reasoner.infer_disjoint_class_axioms(classes_to_infer_over)


.. testcode::

   from org.semanticweb.elk.owlapi import ElkReasonerFactory
   reasoner_factory = ElkReasonerFactory()
   reasoner = reasoner_factory.createReasoner(dataset.ontology)

   mowl_reasoner = MOWLReasoner(reasoner)
