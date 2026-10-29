Worked examples
===============

The :math:`T_3` Helly-obstruction ontology
------------------------------------------

Consider the six-axiom propositional :math:`\mathcal{EL}_\bot` theory :math:`T_3`
(Leemhuis & Kutz, NeSy 2025):

.. math::

   \begin{align}
   A &\equiv X_1 \sqcap X_2, & B &\equiv X_2 \sqcap X_3, & C &\equiv X_1 \sqcap X_3, \\
   A \sqcap B &\sqsubseteq \bot, & B \sqcap C &\sqsubseteq \bot, & A \sqcap C &\sqsubseteq \bot
   \end{align}

with an additional unconstrained concept name :math:`Z` (as in
``tests/geometric/fixtures/t3.owl``). Every concept name
:math:`A, B, C, X_1, X_2, X_3, Z, \top` is satisfiable: the pairwise conjunctions
:math:`X_1 \sqcap X_2 \equiv A`, :math:`X_2 \sqcap X_3 \equiv B`, and
:math:`X_1 \sqcap X_3 \equiv C` are each satisfiable, whereas the triple
conjunction :math:`X_1 \sqcap X_2 \sqcap X_3` entails :math:`A \sqcap B \sqsubseteq \bot`
and is therefore unsatisfiable.

The script below builds the canonical model, constructs exact interval-union and
convex polyhedral models, runs :func:`~mowl.geometric.helly.helly_check` to
obtain the non-box-representability certificate, verifies that
:class:`~mowl.geometric.boxes.BoxModelBuilder` raises
:class:`~mowl.geometric.errors.NotBoxRepresentable`, and shows why the lossy
bounding-box export :func:`~mowl.geometric.export.to_bounding_boxes` is not a
model:

.. code-block:: python

   import mowl
   mowl.init_jvm("2g")

   from mowl.datasets import PathDataset
   from mowl.geometric import (
       BoxModelBuilder,
       CanonicalModel,
       ConvexPolyhedralModel,
       IntervalUnionModel,
       NotBoxRepresentable,
       helly_check,
       to_bounding_boxes,
   )

   NS = "http://mowl.borg/geo#"
   ds = PathDataset("tests/geometric/fixtures/t3.owl")

   # 1. Canonical model via ELK-style completion
   cm = CanonicalModel.from_dataset(ds)
   assert len(cm.names) == 8  # A, B, C, X1, X2, X3, Z, owl:Thing
   assert cm.entails(NS + "A", NS + "X1") and cm.entails(NS + "A", NS + "X2")
   assert not cm.joint_satisfiable((NS + "X1", NS + "X2", NS + "X3"))

   # 2. 1-D interval-union model is an exact, non-degenerate, strongly faithful model
   ivl = IntervalUnionModel.from_canonical_model(cm)
   assert ivl.is_model()
   assert ivl.is_nondegenerate()
   assert ivl.is_strongly_faithful_on_names()

   # 3. Helly check emits a non-degenerate certificate on (X1, X2, X3)
   report = helly_check(cm)
   assert report.complete is True
   assert report.box_representable_nondegenerate is False
   assert report.box_representable_strong is False
   cert = report.certificates_nondegenerate[0]
   assert (cert.c1, cert.c2, cert.c3) == ((NS + "X1",), (NS + "X2",), (NS + "X3",))
   assert set(cert.pairwise_names) == {NS + "A", NS + "B", NS + "C"}

   # 4. BoxModelBuilder raises NotBoxRepresentable carrying the HellyReport
   try:
       BoxModelBuilder(cm).build()
   except NotBoxRepresentable as exc:
       assert len(exc.report.certificates_nondegenerate) == 1

   # 5. Convex polyhedral model is an exact model in dimension |cm.names| - 1 == 7
   poly = ConvexPolyhedralModel.from_canonical_model(cm)
   assert poly.dimension == len(cm.names) - 1 == 7
   assert poly.is_model()
   assert poly.is_nondegenerate()
   assert poly.is_strongly_faithful_on_names()

   # 6. Bounding-box export of the polyhedral model (lossy: NOT a model)
   bboxes = to_bounding_boxes(poly)


Constructing and exporting a box model on a Helly-satisfiable ontology
----------------------------------------------------------------------

When a propositional ontology has no Helly certificate,
:class:`~mowl.geometric.boxes.BoxModelBuilder` searches for an axis-parallel box
model via mixed-integer linear programming (`scipy.optimize.milp`) over
:math:`d = 1, 2, 4, \dots, d_{\max}` and verifies any solution with
:class:`~mowl.geometric.checker.ModelChecker`. The resulting
:class:`~mowl.geometric.boxes.BoxModel` can be exported to NumPy arrays ordered
by :attr:`mowl.datasets.Dataset.class_to_id` using
:func:`~mowl.geometric.export.boxes_as_tensors`:

.. code-block:: python

   import numpy as np
   from mowl.datasets import PathDataset
   from mowl.geometric import (
       BoxModelBuilder,
       CanonicalModel,
       boxes_as_tensors,
       helly_check,
   )

   ds_ok = PathDataset("tests/geometric/fixtures/helly_ok.owl")
   cm_ok = CanonicalModel.from_dataset(ds_ok)

   report_ok = helly_check(cm_ok)
   assert report_ok.complete is True
   assert report_ok.box_representable_nondegenerate is True

   box_model = BoxModelBuilder(cm_ok, d_max=4, time_limit=30.0).build()
   assert box_model.is_model() and box_model.is_nondegenerate()

   # Export as (lo, hi) corners or as (center, offset) for ELBoxEmbeddings / BoxSquaredEL
   lo, hi = boxes_as_tensors(box_model, ds_ok.class_to_id)
   centers, offsets = boxes_as_tensors(
       box_model, ds_ok.class_to_id, as_center_offset=True
   )

   # Note: BoxModelBuilder minimises total box width sum(U - L), so returned boxes
   # frequently have lo == hi (offsets == 0). If seeding a gradient-based trainer
   # that requires positive box volume, clamp offsets from below before training:
   offsets_with_volume = np.maximum(offsets, 1e-3)
