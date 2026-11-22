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
:class:`~mowl.nn.el.boxsel.module.BoxSELModule`, a subclass of the BoxEL model and module.
The module is a :class:`~mowl.representation.ComposedGeometry` made of the parts that the
builder ``boxsel`` (``mowl/representation/library/boxrel_methods.py``) names: boxes stored as
a lower corner and a log-width, an affine map per relation, the smoothed volume ratio as the
measure, the objective of BoxEL with the form "conditional" added, and the two penalties
:math:`L_{loc}` and :math:`L_{vol}`. Any of them can be replaced (see :doc:`/representation/combine`);
the hand-written module that this one replaced is kept as ``mowl.baselines.nn.el.boxsel.module``
and the tests compare the two number by number.


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
ontology) and the paper's datasets (see below). See the catalogue card for the details,
including the volume regulariser of Eq. 10, which is off by default.

Published results and what can be reproduced
--------------------------------------------

The evaluation of the paper is implemented; its data are not available. The paper builds three
Statistical EL ontologies (Person: 135 concepts, 6 roles, 325,600 axioms; Country: 30 concepts,
1 role, 26,974 axioms; Hybrid: 12 concepts, 18 roles, 7,492 axioms) from YAGO3 by counting
(Algorithm 1 of its appendix), but links neither code nor data, and states the selection of
concepts and roles only partly (the roles of Person and Country are not named). They therefore
cannot be rebuilt exactly, and the published numbers cannot be reproduced in mOWL.

What mOWL provides, so that the experiment can be run on an ontology built the same way:

* :func:`~mowl.datasets.builtin.sel_conditionals.conditionals_from_assertions` is Algorithm 1
  (the conditionals ``(B|A)``, ``(B|A1 ⊓ A2)``, ``(B|∃r.A)``, ``(∃r.A|B)`` with their
  statistical proportions, from class and role assertions);
* :class:`~mowl.datasets.builtin.sel_conditionals.SELConditionalDataset` reads such a set
  (a JSON file) and holds out 30 percent of the conditionals of the most general concept as
  queries;
* the regime ``sel-pmp-intervals``
  (:class:`~mowl.evaluation.sel_probability.SELProbabilityRegime`) computes for every query
  the interval entailed by the probabilistic modus ponens from its two premises and scores the
  estimate of the model: soundness accuracy, soundness error, approximation gap and the mean
  absolute error of the embedded probabilities; the fixed and random estimators of the paper are
  reported next to it. ``evaluate_ensemble`` takes the minimum and maximum over several
  embeddings (the paper uses 60 seeds); the KDE baseline is not implemented;
* the benchmarks ``boxsel-sel-person``, ``boxsel-sel-country`` and ``boxsel-sel-hybrid`` read
  ``<data_root>/sel_conditionals/<name>.json``; they cannot run until you place such a file
  there; the published soundness accuracy, soundness error and approximation gap of BoxSEL
  (Person 89.3 percent, 0.013, 0.233; Country 95.6 percent, 0.004, 0.241; Hybrid 89.7 percent,
  0.020, 0.293; arXiv:2407.11821, table ``tab:main``) and the recipes (the paper's best
  hyperparameters) are recorded.

.. note::

   The published results are *not reproducible in mOWL* because the ontologies are not released.
   Numbers obtained on an ontology you build are not comparable with them. The appendix of the
   paper describes the premises of the evaluation in two ways (main text: ``(D|C)`` and
   ``(E|C ⊓ D)``; appendix pseudocode: ``(A|Q1)`` and ``(Q2|A)`` with other bounds); mOWL
   follows the main text.

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
