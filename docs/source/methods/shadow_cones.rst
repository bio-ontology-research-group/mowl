Shadow cones
============

:class:`~mowl.models.ShadowCones` implements the shadow cones of Yu, Liu, Tseng and De Sa
[yu2023shadow]_ for the subclass order of an ontology. The catalogue key is ``shadow_cones``.

Geometry
--------

A light source casts shadows of the objects that represent the classes; :math:`C \sqsubseteq D`
holds when the shadow of :math:`C` is inside the shadow of :math:`D`, which is transitive
because inclusion is (Theorem 3.1). The shadow condition is reduced to point-in-cone
membership. mOWL has the four constructions of the paper (``formulation``):

* ``"umbral-half-space"``: every class is a hyperbolic ball of radius :math:`r` around its
  point, the light source is a point at infinity of the Poincaré half-space; the cone has the
  fixed aperture :math:`\arctan\sinh r`. It is the best performing one in the paper.
* ``"penumbral-half-space"``: points without volume, the light source is the horosphere
  :math:`x_n = e^h` (``horosphere_height``); half-aperture :math:`\arcsin(u_n / e^h)`.
* ``"penumbral-ball"``: a ball light source of radius :math:`r` at the origin of the Poincaré
  ball. Its cones are the entailment cones of Ganea et al. (Theorem 3.2: :math:`K =
  \sinh(r)/2`), see :doc:`hyperbolic_cones`.
* ``"umbral-ball"``: balls around the points, the light source is the origin of the ball.

The score of :math:`C \sqsubseteq D` is the signed hyperbolic distance :math:`d(C,
\mathrm{Cone}(D))` of Section 4: the distance to the apex when :math:`C` lies above the apex
(positive relative altitude) and the signed distance to the cone boundary otherwise, negative
inside the cone. Unlike the angle energy of entailment cones it does not saturate at 0 inside
the cone, so ranking has no ties among contained candidates.

Which axioms and tasks
----------------------

As for :doc:`hyperbolic_cones`: subclass axioms between named classes (``gci0``) and the
subsumption task only; everything else in the ontology is ignored.

Training
--------

The default loss is Eq. 1 of the paper,

.. math::

   -\log \frac{\exp(-\max(E^+, \gamma_2))}{\sum_{k}\exp(\max(\gamma_1 - E^-_k, 0))},

with ``gamma1`` and ``gamma2``. The authors' code trains with a different default (a softmax
cross-entropy that includes the positive in the denominator, ``loss_cross``); that is the
regime ``softmax-energy-contrastive``, which can be attached with
``model.training_regime = SoftmaxEnergyContrastiveLoss(gamma2=0.001)``. Points with positive
altitude use the apex pushed into the cone by ``gamma3`` as in the paper.

Optimisation is Riemannian SGD (:class:`~mowl.nn.order.manifolds.RiemannianSGD`): the
exponential map in the ball, a retraction in the half-space (linear horizontally,
multiplicative in the height).

Differences from the paper
--------------------------

* Curvature is fixed at :math:`-1`; the radius is not trained.
* RiemannianAdam (used for the ball in the paper) is not implemented.
* There is no burn-in phase and the generic loop takes full-batch steps.
* For the ball formulations the printed Appendix F.2 has typographical errors. mOWL uses the
  closed forms derived from the hyperbolic law of sines (aperture :math:`\sin\beta = \sinh r /
  \sinh d(u, O)`), with the relative altitude measured towards the light as in the authors'
  code. Transitivity of all four formulations is checked numerically in the tests; the
  published numbers were not reproduced.

Example
-------

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import ShadowCones

   model = ShadowCones(FamilyDataset(), embed_dim=4, formulation="umbral-half-space",
                       num_neg=4, seed=0)
   model.train(epochs=2)
   print(model.module.class_embed.weight.shape[1])

.. testoutput::
   :options: +ELLIPSIS

   ...4
