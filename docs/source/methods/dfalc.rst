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

* :class:`~mowl.models.dfalc.model.DFALCModel` (catalogue key ``dfalc``), a combination of parts:
  :class:`~mowl.models.ComposedALC` with the method ``"dfalc"``
  (:func:`mowl.representation.library.logic_fuzzy.dfalc`), whose module
  :class:`~mowl.nn.alc.dfalc.module.DFALCModule` is a :class:`~mowl.representation.Geometry`.
  The class model is :class:`~mowl.representation.library.logic_fuzzy.TruthTableClasses`
  (truth values over the domain), the relation model
  :class:`~mowl.representation.library.logic_fuzzy.TruthMatrix`, the operators those of the
  Gödel family, the measure :class:`~mowl.representation.library.logic_fuzzy.MeanViolation`
  (the hierarchical loss) and the objective
  :class:`~mowl.representation.library.logic_fuzzy.FuzzyObjective` (``rule=True`` for the
  rule-based losses). Every part can be exchanged (another operator family, the measure of
  FALCON, ...), see :doc:`/representation/combine`. It reuses the ALC dataset and the
  training loop of :class:`~mowl.models.ComposedALC`, and scores axioms for the joint
  benchmarks (``axiom_scoring="method"``, forms ``gci0``-``gci3``) with the hierarchical loss
  of the candidate inclusion. The hand-written implementation it replaces is kept for
  comparison in :mod:`mowl.baselines.models.dfalc`.
* Axiom forms: ``SubClassOf``, ``EquivalentClasses`` and ``DisjointClasses`` between
  arbitrary :math:`\mathcal{ALC}` class expressions, class assertions and object property
  assertions (a fuzzy assertion with truth value 1; the loss is the distance to 1).
* ``loss_type="rule"`` uses the rule-based losses of the paper: Eq. 9 for inclusions without quantifiers and the losses of normal forms 4-7 for ``L ⊑ ∃r.L``, ``L ⊑ ∀r.L``, ``∃r.L ⊑ L`` and ``∀r.L ⊑ L`` (``threshold`` is the paper's :math:`\alpha`).

Not implemented: the normalisation procedure (not needed for the hierarchical loss) and the neural classifiers
that produce a perceptual grounding (see below for the paper's experiments). The grounding is a table over the domain, so the model
is transductive and meant for small domains. Compared to FALCON
(:class:`~mowl.models.FALCONModel`) there are no entity embeddings, membership networks or
sampled anonymous entities.

Published results and what can be reproduced
--------------------------------------------

The paper has three experiments. Only the first is a pure ontology experiment.

**Masked ABox revision (reproducible, 28 benchmarks).** The ABox of an ontology completed by a
reasoner is the ideal grounding; a fraction (20, 40, 60 or 80 percent) is replaced by random
truth values in the unknown region (0.2, 0.8) to imitate a neural network; the model starts
from this grounding, is trained on the TBox alone, and the *success rate* is the fraction of
TBox axioms that the crisp version of the revised grounding satisfies. The seven ontologies
(Family, Family2, GlycoRDF, Nifdys, Nihss, Ontodm, Sso) are those of the authors' code, which
mOWL downloads at run time
(:class:`~mowl.datasets.builtin.dfalc_revision.DFALCRevisionDataset`); the regime is
``masked-abox-revision``
(:class:`~mowl.evaluation.abox_revision.MaskedABoxRevisionRegime`); the benchmarks are
``dfalc-revision-<ontology>-mask<percent>`` (for example ``dfalc-revision-nifdys-mask20``);
the published success rates of DF-ALC with the hierarchical loss and with the rule-based loss
are recorded for each (arXiv:2211.12006, table ``tab:result1``), with the recipes
``wu2022:hierarchical`` and ``wu2022:rule`` (Adam, learning rate 2e-4, at most 50,000 epochs
as in the code, no early stopping).

.. code-block:: bash

   python -m mowl.benchmarks run dfalc-revision-family-mask20 --method dfalc --recipe wu2022:hierarchical

Caveats: the paper's success rate counts the axioms of the original ontology and its axiom
counts are not those of the released files (mOWL counts the subclass, equivalence and
disjointness axioms it trains on); the released files are a mirror of an anonymous
repository; the paper's Logical Tensor Network baseline is not an mOWL method (its success
rate is in the note of each published row, as is that of the masked grounding).

**Conjunctive query answering (not reproduced).** Precision and recall of the answers of 20
generated conjunctive queries of the forms ``C ⊓ D`` and ``C ⊓ ∃r.D`` against the answers of a
reasoner, on four of the ontologies. The paper reports it only as a figure (no table in the
source); the queries are generated by a Java tool of the authors' code and are not part of
mOWL.

**Semantic image interpretation on PASCAL-Part (not reproducible in mOWL).** The grounding is
the output of a Fast R-CNN object detector on PASCAL-Part images, the part-of relation is
computed from bounding-box overlap, and the ontology (``PascalPartOntology_*``) is in the
authors' repository. The notebooks need the detector outputs and the PASCAL-Part tables that the authors' README
takes from the code of Donadello et al. (2017); these were not obtained or checked, and an image
classifier is not mOWL's business. mOWL can run the ontology side if a grounding is supplied with
:meth:`~mowl.models.dfalc.model.DFALCModel.set_grounding`. For reference, the paper's macro F1
of Fast R-CNN (baseline) and DF-ALC with the rule-based loss when a share of the detector
features is randomised (table ``tab:lowresourse``):

.. list-table::
   :header-rows: 1

   * - Randomised features
     - Vehicle (baseline / DF-ALC)
     - Indoor (baseline / DF-ALC)
     - Animal (baseline / DF-ALC)
   * - 0%
     - 74.71 / 75.97
     - 74.96 / 80.52
     - 74.80 / 75.05
   * - 3%
     - 56.41 / 57.51
     - 68.70 / 71.84
     - 51.77 / 52.73
   * - 5%
     - 48.37 / 49.91
     - 64.03 / 66.90
     - 44.29 / 45.18
   * - 10%
     - 37.31 / 38.45
     - 56.69 / 59.50
     - 32.68 / 33.35

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

The parts of the method can be read back, and exchanged:

.. testcode::

   from mowl.representation import METHODS

   geometry = METHODS["dfalc"](6, 2, 3, embed_dim=4)
   print(type(geometry.class_model).__name__, type(geometry.measure).__name__)
   product = METHODS["dfalc"](6, 2, 3, embed_dim=4, family="product")
   print(type(product.class_model.family).__name__)

.. testoutput::

   TruthTableClasses MeanViolation
   Product
