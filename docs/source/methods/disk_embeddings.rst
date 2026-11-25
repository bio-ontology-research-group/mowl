Hyperbolic disk embeddings
==========================

:class:`~mowl.models.DiskEmbeddings` implements the disk embeddings of Suzuki, Takahama and
Onoda [suzuki2019]_ for the subclass order of an ontology. The catalogue key is
``disk_embeddings``.

Geometry
--------

Every named class is a generalised formal disk :math:`(x, r)`: a centre :math:`x` in a space
with a (quasi-)metric :math:`d` and a real radius :math:`r`, which may be negative.
:math:`C \sqsubseteq D` holds when the disk of :math:`D` contains that of :math:`C`,
:math:`d(x_D, x_C) \le r_D - r_C`. Inclusion is transitive (triangle inequality) and does not
change when a constant is added to all radii (Eq. 4). The score of an axiom is the protrusion

.. math::

   l = d(x_D, x_C) - r_D + r_C \qquad (\text{Eq. 9}),

:math:`\le 0` exactly when the axiom holds. ``geometry`` selects the space: ``"hyperbolic"``
(Lorentz model, the default), ``"spherical"`` or ``"euclidean"``. Order embeddings and
hyperbolic entailment cones are special cases (Theorems 1 and 3), which is why the Euclidean and
spherical variants are included.

Which axioms and tasks
----------------------

As for :doc:`hyperbolic_cones`: subclass axioms between named classes (``gci0``) and the
subsumption task only.

Training
--------

The margin loss of Eq. 10 with :math:`E = l` (Eq. 11): :math:`h_+(l)` for told axioms and
:math:`h_+(\mu - l)` for negatives (child or parent replaced by a random class, 10 per
positive by default). Riemannian SGD updates centres with the exponential map of the geometry
and radii in the Euclidean way (Eqs. 13 and 14, with :math:`\lambda = \nu = 1`).

Differences from the paper
--------------------------

* Ranking uses the signed protrusion :math:`l`, so contained candidates are not tied.
* Generic full-batch training; the paper's spherical model is initialised from Poincaré
  embeddings, which is not done here.
* The polyhedral quasi-metrics of Section 2.2 are not implemented.

Example
-------

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import DiskEmbeddings

   model = DiskEmbeddings(FamilyDataset(), embed_dim=4, num_neg=4, seed=0)
   model.train(epochs=2)
   print(model.module.class_embed.weight.shape[1])

.. testoutput::
   :options: +ELLIPSIS

   ...5
