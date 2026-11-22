Known deviations of published methods
=====================================

While the methods were split into parts, each was compared with the same method rebuilt from
parts. Some differences were defects of mOWL, and they were fixed (the changelog lists them,
and each fix keeps the old behaviour as an option). Others are what the paper or its code does.
A method that is meant to reproduce a paper must keep them, so mOWL keeps them in the named
method and gives a corrected reading as an option or as a part. This page lists those. Where a
paper's text and its code differ, the entry says which one mOWL follows.

The rule behind the choices is on the page :doc:`defaults`.

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
* The scale term of the role chain loss is one-sided: the composed scale must not exceed the scale
  of the target. This is the intended meaning of an inclusion, not an error.

Box2EL and MultiBoxEL
---------------------

For :math:`\exists r.C \sqsubseteq \bot` the positive loss uses the size of the head box and
the negative loss of the DELE paper uses the offset of the filler. Both are published, for
different methods. The part reads the region of :math:`\exists r.C` in both
(``Objective(gci3_bot="exists")``, the default); ``gci3_bot="filler"`` gives the other
reading.

TransBox
--------

The reference formulation follows the authors' code, which has no losses for assertions. For
those, mOWL uses its own formulas. The forms of the code that mOWL calls bottom forms are scored as
ordinary classes in the code, and ``reference_bottom_losses="code"`` (the default) follows
it.

ELBE, ELEmbeddings and Prism
----------------------------

The loss for :math:`\exists r.C \sqsubseteq D` asks that the region of :math:`\exists r.C`
*overlaps* :math:`D`, not that it lies inside. That is the formula of the ELBE paper, and Prism
copies it. The part reads inclusion by default; ``gci3="inclusion"`` of the named methods ELEmbeddings and ELBE
gives the same reading.

EIKE
----

The loss for :math:`C \sqsubseteq D` is a score on ellipsoids that is not containment: it can be zero
while points of :math:`C` lie outside :math:`D`, and the vector length term is signed and can be
negative. Both are as printed in the paper. The ellipsoid part checks a sufficient condition for
containment by default (``EllipsoidMeasure(containment="bounding")``), and the vector part
uses a hinge on the lengths (``CosineOrder(norm_term="hinge")``). The named method ``eike`` sets
``containment="published"`` and ``norm_term="signed"``.

Concept2Box
-----------

For :math:`C \sqsubseteq \exists r.D` the loss asks the box of :math:`D` to lie inside the box
of :math:`C`, the reverse of the axiom, and the weight of the points inside a box makes the
cost non-zero for points that are inside. The first follows the convention of the knowledge
graph triple :math:`(C, r, D)` used in the adaptation, and the second is Equations 8 and 9 of the
paper. We have not yet checked the first against the paper's convention.

Query answering
---------------

* **AConE** wraps the aperture of a cone to :math:`(-\pi, \pi]`, so an aperture can become
  negative. This is the authors' code. ``aperture="clip"`` keeps it in :math:`[0, \pi]`, and
  that is the default of the part.
* **Query2Box** with ``activation="none"`` can produce boxes with negative extent. The
  reference code does the same. ``clamp_extent=True`` clamps the extent at zero.

FALCON
------

The loss :math:`-\log(1 - m + 10^{-10})` can be smaller than zero by at most :math:`10^{-10}`.
This is the same in the original and is negligible. The part's default,
``stabiliser="clamp"``, never goes below zero; the named method uses ``"add"``. ``owl:Thing`` is a
learned class, as in the original (``thing="all"`` makes it the constant true).

Not decided yet
---------------

* The losses of MultiELBE for some bottom forms have shape ``(B,)`` where every other
  composed loss has shape ``(B, 1)``. ELBE has to change first, because MultiELBE with one
  component must equal ELBE.
* The direction of Concept2Box's loss above (not yet compared with the paper).

Behaviours that the named methods still choose
----------------------------------------------

Some defects of mOWL's modules were fixed, and the published reading is still what the named
method and its benchmark recipe use: BoxEL's signed role scale, the raw energy of shadow cones
as the loss of a wrong pair, the ``pi - aperture`` negation of cones in ConE and AConE. They are
described, with the option that selects each, on :doc:`defaults`.
