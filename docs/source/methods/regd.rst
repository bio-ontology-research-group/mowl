RegD
====

RegD [yang2025regd]_ replaces the inclusion energy between two regions by the sum of a
*boundary dissimilarity* and a *depth dissimilarity*. The catalogue key is ``regd``; its
guarantees, with sources and the tests that check them, are on the card in the
:doc:`method catalogue <catalogue>`. mOWL provides two models,
:class:`~mowl.models.regd.RegDELBE` (boxes, RegD in place of the ELBE inclusion energy) and
:class:`~mowl.models.regd.RegDELEm` (balls, RegD in place of the ELEm inclusion energy).

Energy
------

For a parent region :math:`u` (the region of :math:`D` in :math:`C \sqsubseteq D`) and a child
region :math:`v`,

.. math::

   E(u, v) = d_{bd}(u, v) + \lambda\, d_{dep}(u, v).

* Balls: :math:`d_{bd} = \lVert c_u - c_v \rVert_2 + r_v - r_u` and
  :math:`d_{dep} = (\lVert c_u - c_v \rVert_p^p + |r_u - r_v|^p) / (r_u r_v)`, with
  :math:`p = 2`.
* Boxes: with :math:`a = |c_u - c_v| + o_v - o_u`, :math:`d_{bd} = \max_i a_i` if
  :math:`a \le 0` and :math:`\lVert \max(a, 0) \rVert_2` otherwise;
  :math:`d_{dep} = (\lVert c_u - c_v \rVert_p^p + \lVert o_u - o_v \rVert_p^p) /
  (\lVert o_u \rVert \lVert o_v \rVert)`, with :math:`p = 1`.

:math:`d_{bd} \le 0` exactly when the child is inside the parent. :math:`d_{dep}` grows
without bound as the regions shrink, which gives room for many siblings and many levels in
few dimensions (Theorems 1 and 2 of the paper). The functions are in
:mod:`mowl.nn.el.regd.dissimilarity`.

Normal forms
------------

RegD only changes the *inclusion* energy; the host model keeps everything else.

* ``gci0`` :math:`C \sqsubseteq D`: :math:`E(D, C)`.
* ``gci2`` :math:`C \sqsubseteq \exists r.D`: :math:`E(D, C + v_r)`.
* ``gci3`` :math:`\exists r.C \sqsubseteq D`: :math:`E(D, C - v_r)`.
* ``gci1`` :math:`C_1 \sqcap C_2 \sqsubseteq D`: :math:`E(D, C_1 \cap C_2)` plus ELBE's
  non-emptiness term (boxes only). With balls, the ELEm loss is kept.
* Bottom forms and assertions keep the host losses.

``gci0`` and ``gci2`` are trained with the built-in contrastive loss of the paper
(Eq. 10) with 10 sampled negatives per positive, which push a wrong superclass or
filler outside the parent. The other forms minimise :math:`\max\{E, \gamma_1\}`.
The *recipes* of the ``regd-*`` benchmarks instead train with the loss of the authors'
code (regime ``regd-code-training``): a softmax cross-entropy over the energies of one
positive and its 10 negatives (:class:`~mowl.training.SoftmaxEnergyContrastiveLoss`),
and only on the normal form of the task -- ``gci0`` for the inference benchmarks,
``gci2`` for the prediction benchmarks, where the code also constrains the reversed
inclusion (``flip_gci2``). Every other normal form is trained on its *raw* (not
rectified) RegD energy, with squared error for boxes and a plain mean for balls, as the
code's ``Elbe`` and ``Elem`` do; the regime also initialises the regions as the code
does, and the ball model of the recipes uses softplus radii (``radius="softplus"``).

RegD speaks about hierarchies. The faithfulness result of the paper (every partial order has a
faithful embedding in :math:`\mathbb{R}^2`) says nothing about conjunction or disjointness,
and uses convex polygons, not boxes or balls.

Differences from the paper
--------------------------

See the class docstring of :class:`~mowl.models.regd.RegDELBE` and the card notes. The main
ones: Eq. 9 adds the depth term to every positive pair while the authors' code adds it only to
pairs that are not contained (``depth_on_containment=False`` gives that variant); the
negatives replace the superclass or filler, as in the code; the loss of the code is a
softmax cross-entropy over the full energies, not Eq. 10, and it is applied to one normal
form per task; the prediction task is evaluated with the code's own rules (regime
``regd-sampled-f1``: rectified boundary dissimilarity as the score, threshold tuned on
validation, and -- for Table 4 -- the positive score the code reads from the role
column: the score of the class whose reference index equals the role's index, e.g.
*biological_process* for every GO ``part_of`` query; see the regime card).

Example
-------

.. testcode::

   import torch as th
   from mowl.datasets.builtin import FamilyDataset
   from mowl.models.regd import RegDELBE, RegDELEm

   dataset = FamilyDataset()
   model = RegDELBE(dataset, embed_dim=5, learning_rate=0.01)
   model.train(epochs=5)
   model.eval_gci_name = "gci0"
   pairs = th.tensor([[0, 1], [1, 0]])
   with th.no_grad():
       energy = model.eval_method(pairs)   # lower: more likely C subclass of D
   assert energy.shape == (2, 1)

   ball_model = RegDELEm(dataset, embed_dim=5, learning_rate=0.01)
   ball_model.train(epochs=5)

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...
