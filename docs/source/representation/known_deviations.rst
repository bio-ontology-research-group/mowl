Known deviations of published methods
=====================================

Some things a publication or its code does are not what one would expect: a loss that is
zero while the axiom is false, a reading that is the reverse of the axiom, a margin that
works the other way round. mOWL keeps one rule for them. **A part defaults to the correct
reading (a cost of zero exactly when the relation holds). A named method keeps its authors'
reading, as an option that its builder or recipe selects, and this page lists every such
item with the option or the part that gives the corrected reading.** Where a paper's text and
its code differ, the entry says which one mOWL follows. The rule and the changed defaults are
on the page :doc:`defaults`.

An item is on this page only because it is the published behaviour of a method and was kept
on purpose. Defects of mOWL itself were fixed in the code and are in the changelog.

RegD
----

The loss for a pair that should *not* be included is non-inclusion: the boundary distance of
the pair is pushed above a margin (``gamma2``). It is not disjointness. With the default
``gamma2=0`` the loss is almost always zero, so negatives hardly train anything. This is Equation 10 of the paper.
For a negative loss that always acts, raise ``gamma2``, or combine the measure ``"boundary and
depth"`` with the default objective in a free combination, whose negative reading is
disjointness.

BoxEL and BoxSEL
----------------

* The loss for :math:`C \sqcap D \sqsubseteq \bot` is the volume of the intersection divided by
  the product of the two volumes. Scaling all axes changes it, so it is not scale invariant. This is the
  reading of the released code, and a corrected form is ambiguous. ``Objective(gci1_bot="apart")``
  scores disjointness instead.
* The scale term of the role inclusion and role chain losses is one-sided: the scale of the
  sub-role must not exceed that of the super-role. A cost of zero therefore does not mean that
  the two roles are equal maps, and a role is a function, so one function includes another
  only if they are equal. The part ``Affine`` reads the inclusion exactly by default
  (``role_scale="equal"``); the method ``boxel`` keeps the published loss
  (``role_scale="at_most"``), and so does ``BoxEL``.
* The role scale of the released code is signed, so a negative scale turns a box inside out.
  mOWL uses :math:`|s| + \epsilon` (``BoxEL(positive_scale=True)``, the part's
  ``scale="positive"``); ``positive_scale=False`` and ``scale="signed"`` give the released code.
  The chain loss of the released code composes the two maps in the wrong order;
  ``role_chain_composition="as_released"`` gives it, ``"in_order"`` is the default.
* The constructors of ``BoxEL`` and ``BoxSEL`` take ``load_role_axioms`` (default ``False``) so
  that the role inclusion and role chain losses of the module can be trained; before, the
  argument did not exist although the module implements the losses.

Box2EL, MultiBoxEL and Volume MultiBoxEL
----------------------------------------

For :math:`\exists r.C \sqsubseteq \bot` the positive loss uses the size of the head box and
the negative loss of the DELE paper uses the offset of the filler. Both are published, for
different methods. The part reads the region of :math:`\exists r.C` in both
(``Objective(gci3_bot="exists")``, the default); ``gci3_bot="filler"`` gives the other
reading.

MultiBoxEL and MultiELBE read a conjunction or a disjointness of two unions as "every
component of one meets some component of the other". The part ``union of bounds`` is exact by
default: ``meets`` is zero exactly when some pair of components meets
(``meets_reading="some"``); the methods ``multiboxel`` and ``multielbe`` select
``meets_reading="every"``.

MultiBoxEL with ``objective="structural"``: the bump of a defined class is a free parameter
that no definition determines. This is intended and documented on the module.

Volume MultiBoxEL has no distance for :math:`C \sqsubseteq \exists r.D` and
:math:`\exists r.C \sqsubseteq D` in the thesis. mOWL ranks them by the training loss, and many
pairs have distance exactly 1.

TransBox
--------

The reference formulation (``formulation="reference"``) follows the authors' code, which has
no losses for assertions. For those, mOWL uses its own formulas. The code scores the bottom
forms like ordinary classes, and ``reference_bottom_losses="code"`` (the default) follows it;
``"mowl"`` gives the formulas of mOWL's own formulation.

ELBE, ELEmbeddings, Prism and the methods built on them
-------------------------------------------------------

* The loss for :math:`\exists r.C \sqsubseteq D` asks that the region of :math:`\exists r.C`
  *overlaps* :math:`D`, not that it lies inside. That is the formula of the ELBE paper, and
  Prism copies it. The part reads inclusion by default (``Objective(gci3="inclusion")``). The
  named methods ``elem``, ``elbe``, ``emelpp``, ``prism`` and ``multielbe`` select
  ``gci3="overlap"``; passing ``gci3="inclusion"`` to the method gives the corrected reading.
  The model ``ELBE`` has the same switch, ``gci3_form="inclusion"``.
* The loss for :math:`C \sqcap D \sqsubseteq \bot` of ELEmbeddings has a margin that
  tightens the loss (the balls must be apart by :math:`\gamma`), while the margin of the
  inclusion losses loosens it. This is the published form and is documented on the module.
* The bottom losses of Prism for :math:`C \sqsubseteq \bot` and
  :math:`\exists r.C \sqsubseteq \bot` cannot reach zero for a positive margin (the sum of the
  triangle supports is not negative), and the second shrinks :math:`C` itself. They are the
  formulas of the paper and are kept.

ELK-Em
------

The authors' ranking code builds the box of :math:`\exists r.C` by plain interval arithmetic
(it can be non-empty when :math:`C \sqcap R_r` is empty), adds an overlap term to the loss of
:math:`C \sqsubseteq \exists r.D`, and charges the margin for a disjointness row whose
conjunct is fixed to the empty box, a cost no parameter can remove. The default
``variant="paper"`` implements Table 2 of the paper (the empty case stays empty, such a row
costs zero); ``variant="reference"`` keeps the code's behaviour. The argument is in the
module ``ELKEmModule`` and in the model ``ELKEm``.

EIKE
----

The loss for :math:`C \sqsubseteq D` is a score on ellipsoids that is not containment: it can be zero
while points of :math:`C` lie outside :math:`D`, and the vector length term is signed and can be
negative. Both are as printed in the paper. The ellipsoid part checks a sufficient condition for
containment by default (``EllipsoidMeasure(containment="bounding")``), and the vector part
uses a hinge on the lengths (``CosineOrder(norm_term="hinge")``). The named method ``eike`` sets
``containment="published"`` and ``norm_term="signed"``.

@@REST@@
Concept2Box
-----------

* The conditional probability of the paper (Equation 4) is near 1 when the box of the *tail*
  of a triple :math:`(h, r, t)` lies inside the box of its head, so the paper's triple
  :math:`(\text{Artist}, \text{lives\_in}, \text{City})` puts the box of City inside the box of
  Artist, and its example triple :math:`(\text{Singer}, \text{is\_a}, \text{Artist})` puts
  the box of Artist inside that of Singer. The paper does not discuss axioms, so how an
  axiom becomes a triple is a choice of mOWL. :math:`C \sqsubseteq D` is the triple
  :math:`(D, \text{subclass}, C)`: the box of :math:`C` inside that of :math:`D`. In the first
  version :math:`C \sqsubseteq \exists r.D` was the paper's triple :math:`(C, r, D)`, which puts
  the box of :math:`D` inside that of :math:`C`, the reverse of the axiom. That was fixed: the
  default is ``gci2_reading="axiom"`` (the box of :math:`C` inside that of :math:`D` in the
  context of :math:`r`). The paper's triple stays available as ``gci2_reading="triple"``, which
  the recipe of the paper selects.
* The cost of a point inside a box is not zero: the distance of Equation 8 adds
  :math:`\alpha(c)` times the distance of the point to the centre, with
  :math:`\alpha(c) = \alpha\,\mathrm{sigmoid}(1/\mathrm{Vol}(c)) \ge \alpha/2`
  (Equation 9). This is the paper (the inside distance "punishes less if the entity is within the
  box but not close to the center") and is kept in the method ``concept2box``. The part
  ``PointToRegion`` is zero inside the box by default (``alpha=0``).

Query answering
---------------

* **AConE** wraps the aperture of a cone to :math:`(-\pi, \pi]`, so an aperture can become
  negative. This is the authors' code and the default of the model ``AConE``.
  ``aperture="clip"`` keeps it in :math:`[0, \pi]`, and that is the default of the part
  ``ConeRotation``. The authors' negation of a cone is :math:`\pi - a`; for a negative
  aperture it exceeds :math:`\pi`. The method ``acone`` selects it (``signed_negation=True``);
  ``ConE`` and the parts use :math:`\pi - |a|`.
* **Query2Box** with ``activation="none"`` can produce boxes with negative extent. The
  reference code does the same, and so does the model. ``clamp_extent=True`` clamps the extent
  at zero.
* The negation of ConE and AConE is taken per coordinate and is not the true complement of the
  region. The catalogue marks it as approximate.

FaithEL
-------

The negative term of the loss of the authors' code, :math:`-\psi\,\mathrm{mse}(x_a, x_{neg})`,
has no lower bound: with ``neg_sampling=True`` the loss can go to :math:`-\infty` as the
individuals drift apart. The paper text was not available, so the corrected form is mOWL's
choice. The objective part ``AnchoredObjective`` defaults to the hinge
:math:`\psi\,\max(0, m - \mathrm{mse}(x_a, x_{neg}))` (``negative_term="margin"``, margin
``neg_margin=1.0``), which is bounded below by zero; ``negative_term="unbounded"`` gives the
code's term. The class :class:`~mowl.models.FaithEL` selects ``"unbounded"`` (its default), so
its numbers are those of the code; the registered method ``faithel`` of the parts is the
bounded form.

FALCON
------

The rows of the ALC dataset for an assertion :math:`r(a, b)` are ``[b, r, a]`` (object, role,
subject: ``ALCDataset.get_axiom_vector`` takes ``getObject()`` first). The direct code of FALCON
takes the first column as the head of the role network, so it trains
:math:`\mu(e_b + r, e_a)`: the opposite direction of the reading of :math:`\exists r.C`, which is
:math:`\mu(e_x + r, e_y)` for a pair :math:`r(x, y)`. Evidence (a test, trained 300 epochs on
eight assertions :math:`r(a_i, b_i)`): with the reading of the code the mean membership of
:math:`(a_i, b_i)` is 0.06 and of :math:`(b_i, a_i)` is 0.69; with ``abox_head="subject"`` it is
0.60 against 0.18. It was not checked against the authors' own repository, which is not part
of this one. The objective part ``FuzzyObjective`` defaults to ``abox_head="subject"``; the
named method ``falcon`` (and :class:`~mowl.models.FALCONModel`) selects ``"object"`` to
reproduce the code.

The loss :math:`-\log(1 - m + 10^{-10})` can be smaller than zero by at most :math:`10^{-10}`.
This is the same in the original and is negligible. The part's default,
``stabiliser="clamp"``, never goes below zero; the named method uses ``"add"``.
``owl:Thing`` is a learned class, as in the original (``thing="all"`` makes it the constant
true).

Open points
-----------

* Query training: the time of one step grows with the size of the ontology (about 0.34 s per
  step at 100,000 classes after the speed-up; it was 2.2 s).
