Axis-aligned cones for ALC (al-cones)
=====================================

Özçep, Leemhuis and Wolter ([ozcep2020]_, [ozcep2023]_) interpret every
:math:`\mathcal{ALC}` concept as an *axis-aligned cone* of :math:`\mathbb{R}^n`, a product

.. math::

   X = X_1 \times \dots \times X_n, \qquad X_i \in \{\{0\}, \mathbb{R}^+, \mathbb{R}^-, \mathbb{R}\}.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Constructor
     - Interpretation (JAIR Def. 2 and 7)
   * - :math:`\top`, :math:`\bot`
     - :math:`\mathbb{R}^n` and :math:`\{0\}`
   * - :math:`C \sqcap D`
     - :math:`C^I \cap D^I`
   * - :math:`\neg C`
     - polar cone :math:`\{v \mid \langle v, w\rangle \le 0 \text{ for all } w \in C^I\}`
   * - :math:`C \sqcup D`
     - convex union :math:`(C^{I\circ} \cap D^{I\circ})^\circ = \mathrm{conHull}(C^I \cup D^I)`, not the set union
   * - :math:`\exists R.C`
     - :math:`\mathrm{conHull}\{R^T y \mid y \in C^I\}` for a 0/1 incidence matrix :math:`R` over the :math:`2n` half-axes
   * - :math:`\forall R.C`
     - :math:`\neg\exists R.\neg C`
   * - :math:`C \sqsubseteq D`, :math:`C(a)`, :math:`R(a, b)`
     - :math:`C^I \subseteq D^I`; :math:`a^I \ne 0` and :math:`a^I \in C^I`; :math:`b^I \in R\,a^I`

Roles are *not* translations. An al-cone is determined by the half-axes it contains; on
these, intersection, polarity and convex union are intersection, complement and union
(JAIR Prop. 1), so the al-cones of :math:`\mathbb{R}^n` form a Boolean algebra and
:math:`\exists R` is the classical diamond on the frame whose worlds are the half-axes. For
arbitrary convex cones the same operations only give an *orthologic*: distributivity fails
(JAIR Fig. 3(c) and 7(b), [leemhuis2020]_), which is why the papers restrict to al-cones.

Points are *partial*: a point can lie in :math:`C \sqcup D` without lying in :math:`C` or
:math:`D`, and in neither :math:`C` nor :math:`\neg C`. The papers prove that an
:math:`\mathcal{ALC}` ontology is satisfiable iff it has an al-cone model (Prop. 7 for
Boolean ALC, Prop. 12 for full ALC, where the faithful model may be infinite-dimensional),
and give strongly concept-faithful models for Boolean ALC (Prop. 8).

Disjointness and Helly-type arguments
-------------------------------------

Every cone contains the origin, so :math:`C \sqcap D \sqsubseteq \bot` means
:math:`C^I \cap D^I = \{0\}`, not :math:`C^I \cap D^I = \emptyset`. Helly-type arguments
about empty intersections (for example: pairwise intersecting boxes have a common point,
which makes the T3 pattern of Leemhuis and Kutz unrepresentable by boxes) do not carry
over. For *non-trivial* intersections the bound is :math:`2n` in :math:`\mathbb{R}^n`
(our observation: al-cones are subsets of the :math:`2n` half-axes). On T3
(:math:`A \equiv X_1 \sqcap X_2`, :math:`B \equiv X_2 \sqcap X_3`,
:math:`C \equiv X_1 \sqcap X_3`, :math:`A, B, C` pairwise disjoint) the Prop. 7
construction gives a model in :math:`\mathbb{R}^4` with :math:`A, B, C \neq \{0\}` and
pairwise non-trivial :math:`X_i \cap X_j`, while :math:`X_1 \cap X_2 \cap X_3 = \{0\}`.

What mOWL implements
--------------------

:mod:`mowl.geometric.cones` (catalogue key ``alcone-exact``) contains the cone operations,
an exact checker for TBox and ABox axioms (:class:`~mowl.geometric.cones.ALConeChecker`)
and two constructions:

* :meth:`~mowl.geometric.cones.ALConeInterpretation.from_boolean_ontology`: JAIR Prop. 7
  (atoms of the Lindenbaum-Tarski algebra on half-axes) and, with ``faithful=True``, the
  doubled, strongly concept-faithful model of Prop. 8;
* :meth:`~mowl.geometric.cones.ALConeInterpretation.from_finite_model`: an al-cone model
  of full ALC built from any finite classical model (one half-axis per element, roles as
  incidence matrices). This construction is ours; the paper's own construction for full
  ALC is infinite-dimensional in general. For :math:`\mathcal{EL}^{++}` ontologies the
  finite model can be :class:`~mowl.geometric.completion.CanonicalModel`
  (:meth:`~mowl.geometric.cones.ALConeInterpretation.from_ontology`).

:class:`~mowl.models.alcone.model.ALConeModel` (catalogue key ``alcone``) is a learned
version. The papers give no learning algorithm for ontology embedding (JAIR Sec. 7), so the
objective is ours: every concept has two logits per coordinate (one per half-axis, so the
discretised factor is one of :math:`\{0\}, \mathbb{R}^+, \mathbb{R}^-, \mathbb{R}`), roles
have incidence logits and individuals support logits. Complex concepts are evaluated with
Gödel operators (min, max, negation), and every axiom has a hinge loss that, with margin
:math:`\gamma > 0`, is zero only if the discretised interpretation satisfies the axiom
(and is zero if it does and all logits are at least :math:`\gamma` away from 0). Training
uses a smooth version of the operators, annealed to the exact ones. Gradient training can
stop in a local minimum with a few violated axioms, so check the result with
:meth:`~mowl.models.alcone.model.ALConeModel.violations`.

Example
-------

.. testcode::

   import numpy as np
   from mowl.geometric.cones import (ALCone, ALConeInterpretation, Atomic, And, Not, Or,
                                     Exists, GCI, ConceptAssertion, RoleAssertion,
                                     FiniteInterpretation, BOTTOM_CONCEPT)

   A, B, C = Atomic("A"), Atomic("B"), Atomic("C")

   # Cone operations: polar of the quadrant R+ x R+ and the convex union of two half-axes.
   X = ALCone.from_factors("++")
   print(X.polar().factors(), (ALCone.from_factors("+0") | ALCone.from_factors("0+")).factors())

   # JAIR Example 2: TBox {A ⊑ B}, Prop. 7 construction.
   I = ALConeInterpretation.from_boolean_ontology([GCI(A, B)])
   print(I.dim, I.cone(And((A, Not(B)))).is_bottom())

   # A full ALC ontology with a finite model given by hand.
   axioms = [GCI(A, Or((B, C))), GCI(And((B, C)), BOTTOM_CONCEPT),
             GCI(A, Exists("r", Not(A))), ConceptAssertion(A, "a"),
             ConceptAssertion(Not(B), "a"), RoleAssertion("r", "a", "b")]
   model = FiniteInterpretation(2, {"A": {0}, "B": set(), "C": {0}},
                                {"r": {(0, 1)}}, {"a": 0, "b": 1})
   J = ALConeInterpretation.from_finite_model(model, axioms)
   print(J.checker().is_model(axioms))

.. testoutput::

   ('-', '-') ('+', '+')
   2 True
   True

Training the learned model (a few epochs only):

.. testcode::

   from java.util import HashSet
   from mowl.datasets import Dataset
   from mowl.owlapi import OWLAPIAdapter
   from mowl.models.alcone.model import ALConeModel

   ad = OWLAPIAdapter()
   ont = ad.create_ontology("http://example/alcone")
   a_cls, b_cls = ad.create_class("http://example#A"), ad.create_class("http://example#B")
   axs = HashSet()
   axs.add(ad.create_subclass_of(a_cls, b_cls))
   axs.add(ad.create_class_assertion(ad.create_complement_of(b_cls),
                                     ad.create_individual("http://example#x")))
   ad.owl_manager.addAxioms(ont, axs)

   model = ALConeModel(Dataset(ont), dim=2)
   model.train(epochs=5)
   interpretation = model.discretise()      # an exact ALConeInterpretation
   print(type(model.violations()).__name__)

.. testoutput::

   list

The guarantees, with their sources, are listed on the catalogue cards ``alcone`` and
``alcone-exact``.
