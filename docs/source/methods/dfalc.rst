Differentiable fuzzy ALC (DF-ALC)
=================================

DF-ALC (Wu et al., 2022) represents an :math:`\mathcal{ALC}` ontology as a *fuzzy
interpretation over a finite domain*. Every concept name :math:`C` is a vector
:math:`C^I \in [0,1]^{|\Delta|}`, every role name :math:`r` a matrix
:math:`r^I \in [0,1]^{|\Delta| \times |\Delta|}`, and complex concepts use the Gödel
semantics:

.. math::

   (C \sqcap D)^I = \min(C^I, D^I), \quad (C \sqcup D)^I = \max(C^I, D^I), \quad
   (\neg C)^I = 1 - C^I,

   (\exists r.C)^I(a) = \sup_b \min(r^I(a,b), C^I(b)), \quad
   (\forall r.C)^I(a) = \inf_b \max(1 - r^I(a,b), C^I(b)).

An inclusion :math:`C \sqsubseteq D` holds iff :math:`C^I(a) \le D^I(a)` for all
:math:`a \in \Delta`. The hierarchical loss (Eq. 8) is the mean of
:math:`\max(0, C^I(a) - D^I(a))`; when it is zero the grounding is a model of the TBox.
Training therefore *revises* a given grounding (for example the output of an image
classifier) into a model of the ontology. mOWL uses the individuals of the ontology as the
domain (the paper's :math:`\Delta^I = N_I`), optionally extended by anonymous elements.

What mOWL implements
--------------------

* :class:`~mowl.models.dfalc.model.DFALCModel` (catalogue key ``dfalc``), a subclass of
  :class:`~mowl.base_models.EmbeddingALCModel`, with the module
  :class:`~mowl.nn.alc.dfalc.module.DFALCModule`. It reuses the ALC dataset and the generic
  training loop, and scores axioms for the joint benchmarks (``axiom_scoring="method"``,
  forms ``gci0``-``gci3``) with the hierarchical loss of the candidate inclusion.
* Axiom forms: ``SubClassOf``, ``EquivalentClasses`` and ``DisjointClasses`` between
  arbitrary :math:`\mathcal{ALC}` class expressions, class assertions and object property
  assertions (a fuzzy assertion with truth value 1; the loss is the distance to 1).
* ``loss_type="rule"`` uses the rule-based loss of Eq. 9 for inclusions without quantifiers.

Not implemented: the normalisation procedure (not needed for the hierarchical loss), the
rule-based losses of normal forms 4-7, the paper's experiments and the neural classifiers
that produce a perceptual grounding. The grounding is a table over the domain, so the model
is transductive and meant for small domains. Compared to FALCON
(:class:`~mowl.models.FALCONModel`) there are no entity embeddings, membership networks or
sampled anonymous entities.

Example
-------

.. testcode::

   from java.util import HashSet
   from mowl.datasets import Dataset
   from mowl.owlapi import OWLAPIAdapter
   from mowl.models.dfalc.model import DFALCModel

   ad = OWLAPIAdapter()
   ont = ad.create_ontology("http://example/dfalc")
   a, b = ad.create_class("http://example#A"), ad.create_class("http://example#B")
   x = ad.create_individual("http://example#x")
   axs = HashSet()
   axs.add(ad.create_subclass_of(a, b))
   axs.add(ad.create_class_assertion(a, x))
   ad.owl_manager.addAxioms(ont, axs)

   model = DFALCModel(Dataset(ont), anon_domain=2)
   model.train(epochs=50)
   print(model.score_instance("http://example#x", "http://example#A") > 0.5)

.. testoutput::

   True
