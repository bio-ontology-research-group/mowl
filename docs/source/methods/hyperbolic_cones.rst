Hyperbolic entailment cones
===========================

:class:`~mowl.models.HyperbolicCones` implements the hyperbolic entailment cones of Ganea,
Bécigneul and Hofmann [ganea2018]_ for the subclass order of an ontology. The catalogue key
is ``hyperbolic_cones``; guarantees and their tests are on the card in the
:doc:`method catalogue <catalogue>`.

Geometry
--------

Every named class :math:`C` is a point :math:`x_C` of the Poincaré ball :math:`\mathbb{D}^n`
(curvature :math:`-1`), outside a small ball of radius :math:`\epsilon` around the origin, and
has a cone of half-aperture

.. math::

   \psi(x) = \arcsin\Big(K \frac{1 - \lVert x\rVert^2}{\lVert x\rVert}\Big)

pointing away from the origin (Eq. 26; :math:`K = 0.1`, :math:`\epsilon = 2K / (1 +
\sqrt{1 + 4K^2})`). The cones are chosen so that the relation "lies in the cone of" is
transitive (Theorem 4). The axiom :math:`C \sqsubseteq D` is satisfied when :math:`x_C` lies in
the cone of :math:`x_D`. Its score is the energy

.. math::

   E(D, C) = \max\big(0, \Xi(x_D, x_C) - \psi(x_D)\big),

where :math:`\Xi` is the angle at :math:`x_D` between the cone axis and the geodesic to
:math:`x_C` (Eqs. 28 and 33): the angle by which :math:`C` must be rotated to enter the cone.
:math:`E = 0` for every contained pair.

Which axioms and tasks
----------------------

Only subclass axioms between named classes (the ``gci0`` normal form) are used, and the only
inference task is subsumption. Conjunction, existential restrictions, roles, individuals and
disjointness are not represented; axioms that use them are ignored (mOWL warns when it creates
the model). Reflexive axioms and axioms with :math:`\bot` on the left are dropped. The model
declares ``axiom_scoring = "module"`` and ``scored_forms = ("gci0",)``, so it ranks in the joint
``common-*`` subsumption benchmarks.

Training
--------

The loss is the max-margin loss of Eq. 32, :math:`\sum E^+ + \sum \max(0, \gamma - E^-)` with
:math:`\gamma = 0.01`, as means. Negatives corrupt the child and the parent in turn (``num_neg``
per positive, 10 in the paper) and are re-drawn when they are told axioms. Optimisation is
Riemannian SGD with the exponential map (:class:`~mowl.nn.order.manifolds.RiemannianSGD`) and a
projection outside the :math:`\epsilon`-ball, written in PyTorch because ``geoopt`` is not a
dependency. The model uses the generic training loop, so training regimes can be attached to
``model.training_regime`` (for example
:class:`~mowl.training.PerNormalFormMiniBatches` for mini-batches).

Differences from the paper
--------------------------

* The paper starts from 100 epochs of Poincaré embeddings (with a burn-in phase) rescaled by
  0.7; mOWL starts from random points with norm in :math:`[\epsilon, 0.5]`.
* The generic loop takes one full-batch step per epoch (the paper: batches of 10 with a
  learning rate of :math:`10^{-4}` and summed losses), so the default learning rate is larger.
* The Euclidean cones of the paper are not implemented.
* Negatives are checked against told axioms, not against the transitive closure.

Example
-------

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import HyperbolicCones

   model = HyperbolicCones(FamilyDataset(), embed_dim=5, num_neg=4, seed=0)
   model.train(epochs=2)
   print(model.module.class_embed.weight.shape[1])

.. testoutput::
   :options: +ELLIPSIS

   ...5
