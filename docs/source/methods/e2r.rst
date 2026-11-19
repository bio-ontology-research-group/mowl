E2R: quantum-logic subspace embeddings
======================================

Embed2Reason (E2R, [garg2019]_) represents every predicate by a *linear subspace* and
every entity by a vector. The logic of subspaces is quantum logic (Birkhoff and von
Neumann): the operations are

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Constructor
     - Interpretation (Sec. 3)
   * - :math:`\top`, :math:`\bot`
     - the whole space and :math:`\{0\}`
   * - :math:`C \sqcap D`
     - intersection :math:`C \cap D`
   * - :math:`C \sqcup D`
     - span (vector sum) :math:`C + D`, **not** the set union
   * - :math:`\neg C`
     - orthogonal complement :math:`C^\perp`
   * - :math:`C \sqsubseteq D`
     - subspace inclusion
   * - :math:`C(a)`, :math:`R(a, b)`
     - :math:`x_a \in C`; the pair vector :math:`x_{ab} = (x_{aR}, x_{bR})` lies in :math:`R`

For arbitrary subspaces the distributive law fails (three distinct lines :math:`a, b, c`
in :math:`\mathbb{R}^2`: :math:`a \wedge (b \vee c) = a`, but
:math:`(a \wedge b) \vee (a \wedge c) = \{0\}`). E2R therefore restricts all predicates to
*axis-parallel* subspaces, whose projectors commute, so that the law holds
(supplementary material, Theorems 10 and 11). An axis-parallel subspace is a set of
axes, and the lattice is Boolean. Membership is still not classical: :math:`x = (e_1 +
e_2)/\sqrt{2}` lies in :math:`\mathrm{span}(e_1) \sqcup \mathrm{span}(e_2)` but in
neither operand, and in neither :math:`\mathrm{span}(e_1)` nor its complement.

Representation and losses
-------------------------

The paper works in :math:`\mathbb{C}^d` over the field :math:`\mathbb{R}`, i.e. in
:math:`\mathbb{R}^{2d}` (real half, imaginary half). A concept is a relaxed 0/1 indicator
:math:`y \in [0, 1]^{2d}` that uses the real half only, a relation an indicator :math:`z`
with at least one real and one imaginary axis, an entity a unit vector :math:`(x_R, 0)`.
:mod:`mowl.nn.alc.e2r.losses` implements eqs. (1)-(12): structure terms, binarity,
membership (residual of the projection), inclusion :math:`\|y_i \odot (1 - y_j)\|^2`,
conjunction (product), disjunction (max), negation, the universal type restriction
(eq. 9), and the regularisers against *subspace collapse* (sibling orthogonality,
eq. 11, and a dimension gap between parent and child, eq. 12), plus the negated
membership loss of corrupted assertions (3 per positive, Sec. 5).

:class:`~mowl.models.e2r.model.E2R` reads these OWL axioms (named operands only):
class and object property assertions, ``SubClassOf(A, B)`` (also with an intersection
on the right), ``EquivalentClasses`` of a named class with a named class, an
intersection, a union or a complement, ``DisjointClasses``, ``SubObjectPropertyOf``,
``EquivalentObjectProperties``, ``ObjectPropertyRange(R, C)`` and
``SubClassOf(owl:Thing, ObjectAllValuesFrom(R, C))``. The paper's universal type
restriction :math:`\forall R.C` means "every :math:`R`-successor is in :math:`C`", a range
constraint. Existential restrictions have no loss and are listed in
``model.skipped_axioms``.

Ranking for membership queries uses the residual length
:math:`\|(1 - y) \odot x\|` (lower is better, Sec. 5): see
:meth:`~mowl.models.e2r.model.E2R.score_membership`,
:meth:`~mowl.models.e2r.model.E2R.rank_members` (any class expression built from named
classes with :math:`\sqcap`, :math:`\sqcup`, :math:`\neg`) and
:meth:`~mowl.models.e2r.model.E2R.score_role_membership`.
:meth:`~mowl.models.e2r.model.E2R.violations` checks the rounded embedding against the
training axioms with set operations, independently of the losses.

Differences from the paper
--------------------------

Some printed equations contradict the text; mOWL follows the text:

* the second term of eq. (4) is said to vanish for concept memberships, but as printed it
  equals :math:`(\sum_t x_{iR}[t])^2`; it is omitted;
* pair vectors are built as :math:`(x_{pR}, x_{qR})` from the entity vectors, so the
  pair-structure terms of eq. (4) vanish; role assertions also get the unit-length terms
  of both entities;
* concept negation (eq. 8) is evaluated in the real half (the concept universe), otherwise
  it can never be zero together with eq. (2);
* disjointness :math:`A \sqcap B \sqsubseteq \bot` uses eq. (6) with :math:`\bot = \{0\}`
  (the paper has no disjointness loss);
* only the residual of the membership loss is negated for corrupted assertions, so the
  objective is bounded below; indicators are clipped to :math:`[0, 1]` after every step.

The paper does not state initialisation, learning rate, term weights or epochs.

Disjointness, Helly and the T3 pattern (mOWL analysis)
------------------------------------------------------

Every subspace contains the origin, so intersections are never empty, and
:math:`A \sqcap B \sqsubseteq \bot` means :math:`A \cap B = \{0\}`. For axis-parallel
subspaces this is the same as orthogonality (:math:`P_A P_B = 0`, supplementary
Corollary 4); for oblique subspaces orthogonality is stronger. Helly-type arguments about
*empty* intersections of convex sets do not apply. What remains is a statement about
*non-trivial* intersections:

**Proposition (Helly number n).** Let :math:`\mathcal{F}` be a finite family of linear
subspaces of :math:`\mathbb{R}^n`. Then :math:`\bigcap \mathcal{F}` is the intersection of
at most :math:`n` members of :math:`\mathcal{F}`. Hence, if every :math:`n` members
intersect non-trivially, all of them do; the bound is attained by the :math:`n`
coordinate hyperplanes, which are axis-parallel.

*Proof.* Go through :math:`\mathcal{F}` and keep a member only if it strictly lowers the
dimension of the running intersection. Members that are not kept do not change it, so the
final running intersection is :math:`\bigcap \mathcal{F}`; each kept member lowers the
dimension by at least one, starting from :math:`n`, so at most :math:`n` are kept. Any
:math:`n - 1` coordinate hyperplanes contain the remaining axis; all :math:`n` meet in
:math:`\{0\}`. :math:`\square`

**Proposition (T3).** The pattern :math:`A \equiv X_1 \sqcap X_2`,
:math:`B \equiv X_2 \sqcap X_3`, :math:`C \equiv X_1 \sqcap X_3`, with :math:`A, B, C`
pairwise disjoint (meeting in :math:`\{0\}`) and non-trivial, has a subspace model in
:math:`\mathbb{R}^n` iff :math:`n \geq 3`; in E2R this means :math:`d \geq 3`.

*Proof.* In :math:`\mathbb{R}^3` take :math:`X_1 = \mathrm{span}(e_1, e_3)`,
:math:`X_2 = \mathrm{span}(e_1, e_2)`, :math:`X_3 = \mathrm{span}(e_2, e_3)`; then
:math:`A, B, C` are the three axes, pairwise orthogonal (so disjoint in both senses), and
the model is axis-parallel. Three generic planes through the origin work as well (their
pairwise intersections are three distinct lines). In :math:`\mathbb{R}^2`: if two of the
:math:`X_i` were equal, say :math:`X_1 = X_2`, then :math:`C = X_1 \cap X_3 \subseteq A`
and :math:`A \cap C = C \neq \{0\}`; so they are distinct and non-zero. If none is
:math:`\mathbb{R}^2` they are distinct lines and :math:`A = X_1 \cap X_2 = \{0\}`; if one
is, the other two are distinct lines whose intersection (one of :math:`A, B, C`) is
:math:`\{0\}`. :math:`\square`

So, unlike for boxes, T3 is not an obstruction in itself; the analogous obstruction is
dimensional (a family needs :math:`n` members to have trivial intersection while all
smaller subfamilies meet non-trivially). One practical caveat: the sibling-orthogonality
regulariser of eq. (11) asks :math:`X_1 \perp X_2` for the sibling roots
:math:`X_1, X_2`, which contradicts :math:`A \neq \{0\}`; set ``reg_orth=0`` for
ontologies with overlapping siblings. These statements are tested in
``tests/nn/test_e2r_module.py`` (``TestT3``, ``TestNonDistributivity``) and
``tests/models/test_e2r_model.py``.

Benchmarks
----------

The paper reports link prediction on FB15K and WN18 (knowledge-graph benchmarks outside
mOWL's scope) and a reasoning experiment on LUBM1U (Table 1). The LUBM generator is
public, but the experiment cannot be reproduced faithfully: the selection of the 69,628
training triples is not stated, and the eight test queries (members of Professor,
Faculty, Person, Student, Course, Organization, MemberOf, WorksFor) are described only by
name, without their answer sets, candidate sets, or how binary queries are ranked. No
published numbers are recorded in mOWL.

Example
-------

.. testcode::

   from java.util import HashSet
   from mowl.datasets import Dataset
   from mowl.owlapi import OWLAPIAdapter
   from mowl.models.e2r.model import E2R
   from mowl.nn.alc.e2r.module import projector, meet_projector, join_projector, rank

   ad = OWLAPIAdapter()
   ont = ad.create_ontology("http://example/e2r")
   person = ad.create_class("http://example#Person")
   male = ad.create_class("http://example#Male")
   axs = HashSet()
   axs.add(ad.create_subclass_of(male, person))
   axs.add(ad.create_class_assertion(male, ad.create_individual("http://example#bob")))
   axs.add(ad.create_class_assertion(person, ad.create_individual("http://example#eve")))
   ad.owl_manager.addAxioms(ont, axs)

   model = E2R(Dataset(ont), embed_dim=4, learning_rate=0.05, sep_gap=1.0)
   history = model.train(epochs=5)
   print(len(history), model.score_membership("http://example#Person").shape)

   # Quantum logic is not distributive for oblique subspaces (three lines in R^2).
   a, b, c = projector([[1.0], [0.0]]), projector([[0.0], [1.0]]), projector([[1.0], [1.0]])
   print(rank(meet_projector(a, join_projector(b, c))),
         rank(join_projector(meet_projector(a, b), meet_projector(a, c))))

.. testoutput::

   5 (2,)
   1 0

The guarantees, with their sources, are listed on the catalogue card ``e2r``.
