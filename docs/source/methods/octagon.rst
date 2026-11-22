Octagon embeddings (OctagonEL)
==============================

:class:`OctagonEL <mowl.models.octagon.model.OctagonEL>` (the combination of parts called
``octagon``: :doc:`/representation/combine`) represents every concept as a
product of axis-aligned 2-D octagons. The octagons come from Charpenay & Schockaert,
*Capturing Knowledge Graphs and Rules with Octagon Embeddings* (IJCAI 2024,
`arXiv:2401.16270 <https://arxiv.org/abs/2401.16270>`_), where they are **relation**
regions of a knowledge graph embedding. Using them as **concept** regions for
:math:`\mathcal{EL}^{++}` is an adaptation made in mOWL: the octagon geometry is the
paper's, the concept and role design, the losses and the Helly analysis below are ours.

Geometry
--------

A 2-D octagon is the set of points satisfying eight half-plane constraints with fixed
normals :math:`n_k` at angles :math:`k \cdot 45^\circ`:

.. math::

   O(h) = \{(x, y) : x^- \le x \le x^+,\ y^- \le y \le y^+,\
   u^- \le y - x \le u^+,\ v^- \le x + y \le v^+\}
   = \{p : n_k \cdot p \le h_k,\ k = 0, \dots, 7\}.

An axis-parallel rectangle is the special case in which the diagonal bounds are slack. A
concept :math:`C` in :math:`\mathbb{R}^d` (``d`` even) is a product
:math:`O(h_C^{(1)}) \times \dots \times O(h_C^{(d/2)})` with one octagon on each coordinate
pair :math:`(x_{2i}, x_{2i+1})`. It is stored as a centre :math:`c` and eight non-negative
radii :math:`r` per pair, :math:`h_k = n_k \cdot c + |r_k|`, so the centre always belongs to
the region and concept names are never empty (:math:`5d` numbers per concept).

**Tight form** (Proposition 1 of the paper). Each bound is implied by pairs of other
bounds, e.g. :math:`x \le (v^+ - u^-)/2` because :math:`x = ((x + y) - (y - x))/2`. For
every direction there are exactly three pairs of directions that bracket it with an angle
below :math:`180^\circ`; the tight bound is

.. math::

   T(h)_k = \min\big(h_k,\ \min_{n_k = \alpha n_i + \beta n_j,\ \alpha, \beta > 0}
   \alpha h_i + \beta h_j\big).

One pass suffices: for a non-empty octagon :math:`T(h)_k` equals the support function
:math:`\max_{p \in O(h)} n_k \cdot p` (an optimal dual solution of a two-variable linear
programme has at most two non-zero multipliers). Hence :math:`O(T(h)) = O(h)` and
:math:`T` is idempotent. The function :func:`mowl.representation.library.octagon_geometry.tighten`
implements it, and a test checks it against a transcription of the paper's proposition and
against a linear programme.

**Emptiness.** :math:`O(h)` is empty iff some width
:math:`T(h)_k + T(h)_{k+4}` (:math:`k = 0..3`) is negative. If the octagon is non-empty the
widths are non-negative; conversely, an empty set of half-planes in the plane has an
infeasibility certificate with at most three constraints, and solving it for one normal
gives a pair :math:`(i, j)` whose implied bound makes a width negative. A width of exactly
zero means a non-empty octagon with empty interior (a segment or a point).

**Intersection, containment, disjointness.** :math:`O(h_1) \cap O(h_2) = O(\min(h_1, h_2))`
(the paper's Section 3), tightened afterwards. For non-empty :math:`C`,
:math:`C \subseteq D` iff :math:`T(h_C) \le h_D` on every direction (support functions).
:math:`C \cap D = \emptyset` iff :math:`\min(h_C, h_D)` has a negative width. For products,
everything is done factor by factor; a product is empty iff one factor is. Translating an
octagon by :math:`t` adds :math:`n_k \cdot t` to :math:`h_k`.

Axioms and losses
-----------------

Roles are translations, :math:`R^{\mathcal{I}} = \{(x, x + t_R)\}`, as in ELBE and the prism
model. Then :math:`\exists R.C = C - t_R` exactly. Every loss is exact: at margin
:math:`0` it is zero if and only if the geometric condition holds.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Normal form
     - Zero loss iff
   * - :math:`C \sqsubseteq D`
     - :math:`C \subseteq D` (with margin :math:`m`: :math:`T(h_C) + m \le h_D`)
   * - :math:`C_1 \sqcap C_2 \sqsubseteq D`
     - :math:`C_1 \cap C_2 \ne \emptyset` and :math:`C_1 \cap C_2 \subseteq D` (the
       non-emptiness requirement is stronger than the axiom, as in ELBE)
   * - :math:`C_1 \sqcap C_2 \sqsubseteq \bot`
     - smallest width of :math:`C_1 \cap C_2` is at most :math:`-m_\bot`
       (``disjoint_margin``): disjoint for :math:`m_\bot > 0`, empty interior for
       :math:`m_\bot = 0`
   * - :math:`C \sqsubseteq \exists R.D`
     - :math:`C + t_R \subseteq D`
   * - :math:`\exists R.C \sqsubseteq D`
     - :math:`C - t_R \subseteq D` (ELBE and the prism model only require an overlap here,
       which is not sound)
   * - :math:`C \sqsubseteq \bot`, :math:`\exists R.C \sqsubseteq \bot`
     - :math:`C` has empty interior (names are non-empty by construction)
   * - :math:`C(a)`, :math:`R(a, b)`
     - :math:`p_a \in C`; :math:`p_a + t_R = p_b`

Negative samples of :math:`C \sqsubseteq \exists R.D` use the exact non-inclusion loss
:math:`\mathrm{relu}(m^- - \max_k (T(h_{C + t_R}) - h_D)_k)`. Inclusion losses aggregate like
ELBE: on the two axis-parallel facet directions of a factor they equal ELBE's
per-coordinate term :math:`\mathrm{relu}(|c_C - c_D| + o_C - o_D + m)`. On boxes the
inclusion and :math:`\exists`-losses therefore have the same zero set as ELBE's and lie
between 1 and :math:`\sqrt 5` times ELBE's value (tested). Role inclusions and role chains
are not supported: with translations, :math:`R \sqsubseteq S` would force
:math:`t_R = t_S`. The paper composes octagon *relations* in closed form (its Proposition
2), but that requires roles to be octagon relations, which is a different design.

Training
--------

The model uses the generic training loop (full batch, Adam). Two options help exact
hinge losses: ``lr_decay`` (exponential learning-rate decay; with a constant rate Adam
oscillates around equalities such as :math:`A \equiv X_1 \sqcap X_2`) and ``min_width``
(a floor on the widths of all concept names, which stops concepts from collapsing to
near-points; it conflicts with :math:`C \sqsubseteq \bot`). On :math:`T_3` in
:math:`d = 2` from random initialisation (slow test ``test_t3_from_random_initialisation``):
without the floor the largest per-axiom loss falls to about :math:`10^{-5}`, but through
degenerate, near-point regions; with ``min_width=0.02`` the regions stay non-degenerate and
the learned :math:`X_1, X_2, X_3` have no common point, but the definitions keep a residual
of about :math:`10^{-2}` (best of four seeds). The exact model below is therefore not
reached by gradient descent; this is an optimisation gap, not a representation limit.

Helly analysis
--------------

This section is our analysis; the only external results used are Helly's theorem and the
box results of Leemhuis & Kutz (NeSy 2025).

The *Helly number* of a family of sets is the smallest :math:`h` such that every finite
subfamily in which every :math:`h` members intersect has a common point. Helly's theorem
(1923): convex sets in :math:`\mathbb{R}^m` have Helly number at most :math:`m + 1`, so
planar convex sets have Helly number 3. Axis-parallel boxes have Helly number 2, which is
why :math:`T_3 = \{A \equiv X_1 \sqcap X_2,\ B \equiv X_2 \sqcap X_3,\ C \equiv X_1
\sqcap X_3,\ A, B, C \text{ pairwise disjoint}\}` has no box model in which the names are
non-empty (Leemhuis & Kutz 2025): the three :math:`X_i` intersect pairwise, hence have a
common point, which lies in :math:`A \cap B`.

**Proposition 1 (upper bound).** Products of :math:`P` planar convex sets in
:math:`\mathbb{R}^{2P}`, in particular products of 2-D octagons in any even dimension, have
Helly number at most 3.

*Proof.* For products, :math:`\bigcap_i \prod_p O_i^{(p)} = \prod_p \bigcap_i O_i^{(p)}`, so
a family of products intersects iff in every factor :math:`p` the factor sets intersect.
If every three products intersect, then in every factor every three planar convex sets
intersect, so by Helly's theorem each factor family has a common point :math:`q_p`, and
:math:`(q_1, \dots, q_P)` lies in all the products. :math:`\square`

The same argument is the core of the paper's proof of its Proposition 7, for
coordinate-wise relation regions.

**Proposition 2 (the bound is attained).** In :math:`\mathbb{R}^2` let :math:`X_1 = [0, 1]
\times [0, 0.1]`, :math:`X_2 = [0, 0.1] \times [0, 1]` and :math:`X_3 = \{(x, y) \in [0,
1]^2 : 0.9 \le x + y \le 1\}`. They are octagons, intersect pairwise, and
:math:`X_1 \cap X_2 \cap X_3 = \emptyset` (on :math:`X_1 \cap X_2`, :math:`x + y \le 0.2`).
Taking the same box in all other factors gives the same in every even dimension, so the
Helly number of products of 2-D octagons is exactly 3.

Consequences, all proven above:

* :math:`T_3` has an exact octagon model in :math:`d = 2`: the :math:`X_i` above,
  :math:`A = X_1 \cap X_2` and so on. The disjointness gap is :math:`0.7`, so all losses are
  zero even with ``disjoint_margin = 0.05`` (test
  ``test_t3_exact_zero_loss_configuration``; an independent sampling check verifies every
  axiom pointwise).
* :math:`T_4 = \{A_i \equiv \bigsqcap_{j \ne i} X_j \ (i = 1..4),\ X_1 \sqcap X_2 \sqcap
  X_3 \sqcap X_4 \sqsubseteq \bot\}` has no octagon-product model with non-empty
  :math:`A_i`, in any dimension: non-empty :math:`A_i` make every three :math:`X_j`
  intersect, so all four share a point. More generally, no octagon-product model can make
  :math:`k \ge 4` concepts intersect three by three without a common point. A classical
  model exists (four elements, :math:`X_j = \{a_i : i \ne j\}`). The test
  ``test_t4_definitions_force_a_common_point`` sets random :math:`X_j` and the defined
  classes to their definitions and checks that the disjointness loss is then never below its
  margin.
* Every box model with translation roles in :math:`\mathbb{R}^n` is an octagon-product
  model in :math:`\mathbb{R}^n` (:math:`n` even) or :math:`\mathbb{R}^{n+1}` (add a constant
  coordinate). Prisms (Federico, Leemhuis & Kutz, DL 2026) also have Helly number 3 by the
  same argument, but need up to two extra dimensions to contain a box model (their Theorem
  9). Octagon products and prisms are incomparable as shapes (an equilateral triangle is
  not an octagon).

What we do not claim: we do not characterise which ontologies have octagon-product models.
Whether every ontology that avoids the Helly-3 obstruction has one is open.

Example
-------

The exact :math:`T_3` configuration and its losses:

.. testcode::

   import torch as th
   from mowl.datasets.builtin import FamilyDataset
   from mowl.representation.library import octagon_geometry as G
   from mowl.nn.el.octagon.module import OctagonELModule

   x1 = G.from_box(th.tensor([0.5, 0.05]), th.tensor([0.5, 0.05]))
   x2 = G.from_box(th.tensor([0.05, 0.5]), th.tensor([0.05, 0.5]))
   x3 = th.tensor([1.0, 1.0, 1.0, 1.0, 0.0, -0.9, 0.0, 1.0])  # 0.9 <= x + y <= 1
   regions = [x1, x2, x3, G.intersect(x1, x2), G.intersect(x2, x3), G.intersect(x1, x3)]

   # classes 0..5 = X1, X2, X3, A, B, C; class 6 plays the role of owl:Nothing
   module = OctagonELModule(7, 1, embed_dim=2, margin=0.0, disjoint_margin=0.05)
   for i, h in enumerate(regions):
       module.set_class_bounds(i, h.unsqueeze(0))

   gci0 = th.tensor([[3, 0], [3, 1], [4, 1], [4, 2], [5, 0], [5, 2]])
   gci1 = th.tensor([[0, 1, 3], [1, 2, 4], [0, 2, 5]])
   gci1_bot = th.tensor([[3, 4, 6], [4, 5, 6], [3, 5, 6]])
   total = sum(module(g, n).sum() for g, n in
               [(gci0, "gci0"), (gci1, "gci1"), (gci1_bot, "gci1_bot")])
   print(f"{float(total):.4f}")

.. testoutput::

   0.0000

Training the model on a dataset:

.. testcode::

   from mowl.models.octagon.model import OctagonEL

   model = OctagonEL(FamilyDataset(), embed_dim=4, learning_rate=0.01)
   model.train(epochs=2, validate_every=10)
   print(model.count_class_parameters())

.. testoutput::
   :options: +ELLIPSIS

   ...20

References
----------

* Charpenay, V., & Schockaert, S. (2024). Capturing Knowledge Graphs and Rules with Octagon
  Embeddings. IJCAI 2024. https://arxiv.org/abs/2401.16270 (code:
  https://github.com/vcharpenay/uvxy)
* Leemhuis, M., & Kutz, O. (2025). Understanding the Expressive Capabilities of Knowledge
  Base Embeddings under Box Semantics. NeSy 2025, PMLR 284.
* Federico, G., Leemhuis, M., & Kutz, O. (2026). Introducing Prism Embeddings: A New Family
  of Geometric Ontology Embeddings. DL 2026, CEUR-WS Vol-4230.
* Helly, E. (1923). Über Mengen konvexer Körper mit gemeinschaftlichen Punkten.
  Jahresbericht der Deutschen Mathematiker-Vereinigung 32, 175–176.
