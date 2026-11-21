BoxSEL: box embeddings of Statistical EL
========================================

Statistical EL (SEL, Peñaloza and Potyka) adds conditional statements :math:`(D \mid C)[l, u]`
to :math:`\mathcal{EL}`: between :math:`l` and :math:`u` of the elements of :math:`C` are in
:math:`D`. BoxSEL (Zhu et al., 2024) embeds a SEL TBox with the boxes of BoxEL and replaces
cardinalities by volumes, so that a conditional holds when

.. math::

   l \cdot \mathrm{Vol}(C) \;\le\; \mathrm{Vol}(D \cap C) \;\le\; u \cdot \mathrm{Vol}(C).

The conditional probability of an embedding is the volume ratio
:math:`\mathrm{Vol}(D \cap C)/\mathrm{Vol}(C)`. Embeddings with different seeds give a lower
and an upper estimate of the entailed interval (minimum and maximum over the embeddings).

What mOWL implements
--------------------

:class:`~mowl.models.boxsel.model.BoxSEL` (catalogue key ``boxsel``) with the module
:class:`~mowl.nn.el.boxsel.module.BoxSELModule`, a subclass of the BoxEL model and module:

* The ontology gives the deterministic axioms (all :math:`\mathcal{EL}^\bot` normal forms and
  assertions), trained with the BoxEL losses and scored for the joint benchmarks as BoxEL.
* The probabilistic conditionals are a Python argument, because OWL cannot express them:
  ``Conditional(conclusion, premise, lower, upper)``. Concepts are class IRIs,
  ``("and", e1, e2, ...)`` and ``("some", role_iri, e)``; this includes the four forms of the
  paper's prototype. They are trained with the volume losses (softplus volumes, optionally with
  the paper's cooling) through the generic EL training loop.
* ``BoxSEL.probability(D, C)`` is the exact volume ratio, ``conditional_errors`` the distance of
  the embedded probabilities to their intervals, ``BoxSEL.probability_interval`` the estimate
  over several embeddings.

Not implemented: the normalisation of conditionals over complex concepts (define a name in the
ontology), the paper's datasets and evaluation. See the catalogue card for the details,
including the volume regulariser of Eq. 10, which is off by default.

Example
-------

.. testcode::

   from java.util import HashSet
   from mowl.datasets import Dataset
   from mowl.owlapi import OWLAPIAdapter
   from mowl.models import BoxSEL
   from mowl.models.boxsel import Conditional

   ad = OWLAPIAdapter()
   ont = ad.create_ontology("http://example/boxsel")
   applicant = ad.create_class("http://example#Applicant")
   dept_a = ad.create_class("http://example#DeptA")
   axs = HashSet()
   axs.add(ad.create_subclass_of(dept_a, applicant))
   ad.owl_manager.addAxioms(ont, axs)

   # 20 to 25 percent of the applicants applied to department A
   conditionals = [Conditional("http://example#DeptA", "http://example#Applicant", 0.2, 0.25)]
   model = BoxSEL(Dataset(ont), conditionals=conditionals, embed_dim=4, learning_rate=0.05)
   model.train(epochs=20)
   p = model.probability("http://example#DeptA", "http://example#Applicant")
   print(0.0 <= p <= 1.0)

.. testoutput::
   :options: +ELLIPSIS

   ...True
