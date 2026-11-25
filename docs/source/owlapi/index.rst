OWLAPI
=========


mOWL interfaces the OWLAPI by using JPype to connect to the Java Virtual Machine. JPype allows to access Java objects and methods from a Python script. For example, we can use the OWLAPI from Python:

.. testcode::
   
   import mowl
   mowl.init_jvm("10g")
   from org.semanticweb.owlapi.apibinding import OWLManager

   manager = OWLManager.createOWLOntologyManager()
   ontology = manager.createOntology()

   ontology.getClassesInSignature()

.. note::

   Notice that accessing to Java objects is only possible after starting the JVM


OWLAPI version
--------------------------------

mOWL packages OWLAPI 5.5.1, with the reasoners ELK 0.6.0 and HermiT 1.4.5. The methods of
OWLAPI 4 that return sets (``getAxioms()``, ``getClassesInSignature()``,
``getOperands()``, ...) are still there, next to the stream methods of OWLAPI 5
(``axioms()``, ``classesInSignature()``, ...). A Java stream is not a Python iterable:
iterate over its ``iterator()``. Lists that OWLAPI returns
(``getOperandsAsList()``) are read-only, and factory methods that take several class
expressions need Java arguments or a ``java.util.ArrayList``, not a Python list.

.. testcode::

   from java.util import ArrayList
   from org.semanticweb.owlapi.model import IRI
   from org.semanticweb.owlapi.search import EntitySearcher

   factory = manager.getOWLDataFactory()
   named = [factory.getOWLClass(IRI.create("http://example.org/" + n)) for n in "AB"]
   both = factory.getOWLObjectIntersectionOf(ArrayList(named))
   operands = list(both.getOperandsAsList())          # a Python copy that can be changed
   annotations = list(EntitySearcher.getAnnotations(named[0], ontology).iterator())
   print(len(operands), len(annotations), len(list(ontology.axioms().iterator())))

.. testoutput::

   2 0 0

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

mOWL provides some shortcuts for performing reasoning over ontologies. This works by wrapping OWL Reasoners that are instances of ``org.semanticweb.owlapi.reasoner.OWLReasoner``. The following example shows how to obtain inferences from an ontology.

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


The previous example uses the HermiT reasoner. We can also use ELK:

.. testcode::

   from org.semanticweb.elk.owlapi import ElkReasonerFactory
   reasoner_factory = ElkReasonerFactory()
   reasoner = reasoner_factory.createReasoner(dataset.ontology)

   mowl_reasoner = MOWLReasoner(reasoner)
