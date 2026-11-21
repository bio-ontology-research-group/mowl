Temporal al-cone models (LTL ALC)
=================================

Leemhuis ([leemhuis2024temporal]_) extends the axis-aligned cone semantics of
:doc:`alcone` to the role-free temporal description logic :math:`LTL_{\mathcal{ALC}}`
(next :math:`\circ`, until :math:`\mathcal{U}`, eventually :math:`\Diamond`, always
:math:`\Box`; flexible concepts, rigid constants, constant domain). Every time point has its
own al-cone interpretation in the same :math:`\mathbb{R}^n`; an individual is a *ray*, so it
keeps its identity over time, and the paper proves that an ontology is satisfiable iff it has
such a geometric model (Theorem 2). The paper gives a construction and a proof, no learning
procedure, and mOWL implements the construction (catalogue key ``temporal_cones``, status
*constructive*) in :mod:`mowl.geometric.temporal_cones`.

What is implemented
-------------------

* Syntax: :class:`~mowl.geometric.temporal_cones.Next`,
  :class:`~mowl.geometric.temporal_cones.Until`, ``Eventually``, ``Always`` and the ALC
  classes of :mod:`mowl.geometric.cones`; or strings (``"~M & F M <= F (C U M)"``).
  Temporal axioms have no OWL syntax, so they are supplied as Python objects or strings,
  separately from the geometry.
* :class:`~mowl.geometric.temporal_cones.TemporalALConeInterpretation`: concepts are
  :class:`~mowl.geometric.cones.ALCone` objects per time point. :math:`\sqcap`,
  :math:`\neg`, :math:`\sqcup` are intersection, polar and convex union of the same time
  point, :math:`\circ C` is the cone of the next time point, :math:`C\,\mathcal{U}\,D` is
  the convex union of :math:`D(m) \sqcap \bigsqcap_{i\le k<m} C(k)`. The time line is
  ultimately periodic (Theorem 1): :math:`L` time points, the last followed by
  ``loop_start``. A GCI holds iff :math:`C(t) \subseteq D(t)` at every time point; an
  assertion :math:`C(a)` is evaluated at time 0 (our convention).
* The construction of Theorem 2: from a finite ultimately periodic classical model
  (:class:`~mowl.geometric.temporal_cones.LassoModel`) each distinct trace of an element
  becomes a half-axis, atoms are the cones of the half-axes of the traces that have them,
  individuals are rays on their trace's half-axis
  (:meth:`~mowl.geometric.temporal_cones.TemporalALConeInterpretation.from_lasso_model`).
* :func:`~mowl.geometric.temporal_cones.find_lasso_model`: a bounded search for such a
  model (the logic has no roles, so only single traces have to be checked).

Differences from the paper
--------------------------

* The paper writes until as a set union of al-cone pieces. With a set union,
  :math:`B \sqsubseteq \Diamond A` can fail geometrically although it holds on every
  element, so mOWL uses the convex union inside until; every concept is then an al-cone at
  every time point and the construction is correct (this is tested against an independent
  classical evaluator).
* Satisfiability is only searched up to a prefix and period length; the paper gives no
  bound. Roles are not covered (neither in the paper). There is no training.

Example
-------

.. testcode::

   from mowl.geometric.temporal_cones import (
       TemporalALConeInterpretation, TemporalOntology, parse_concept)

   # Non-members of the EU that eventually become members are candidates first (Example 3).
   onto = TemporalOntology.parse(
       tbox=["~M & F M <= F (C U M)"],
       abox=["(~M & F M)(turkey)", "M(germany)"])
   model = TemporalALConeInterpretation.from_ontology(onto, max_prefix=1, max_period=2)
   print(model.is_model(onto), model.dim >= 1)
   print(model.instance_of("turkey", parse_concept("F (C U M)"), 0))

.. testoutput::

   True True
   True
