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

.. testcode::

   import pathlib

   import mowl
   mowl.init_jvm("2g")  # no-op if the JVM is already running

   FIXTURES = pathlib.Path(mowl.__file__).resolve().parents[1] / "tests" / "geometric" / "fixtures"

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
   ds = PathDataset(str(FIXTURES / "t3.owl"))

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

When a propositional ontology has no non-degenerate Helly certificate
(it is *obstruction-free*; ``box_representable_nondegenerate`` being ``True``
means exactly this and does not imply that a box model exists),
:class:`~mowl.geometric.boxes.BoxModelBuilder` searches for an axis-parallel box
model via mixed-integer linear programming (`scipy.optimize.milp`) over
:math:`d = 1, 2, 4, \dots, d_{\max}` and verifies any solution with
:class:`~mowl.geometric.checker.ModelChecker`. The resulting
:class:`~mowl.geometric.boxes.BoxModel` can be exported to NumPy arrays ordered
by :attr:`mowl.datasets.Dataset.class_to_id` using
:func:`~mowl.geometric.export.boxes_as_tensors`:

.. testcode::

   import pathlib

   import mowl
   mowl.init_jvm("2g")  # no-op if the JVM is already running

   FIXTURES = pathlib.Path(mowl.__file__).resolve().parents[1] / "tests" / "geometric" / "fixtures"

   import numpy as np
   from mowl.datasets import PathDataset
   from mowl.geometric import (
       BoxModelBuilder,
       CanonicalModel,
       boxes_as_tensors,
       helly_check,
   )

   ds_ok = PathDataset(str(FIXTURES / "helly_ok.owl"))
   cm_ok = CanonicalModel.from_dataset(ds_ok)

   # The Helly check finds no obstruction ("obstruction-free"). This does NOT
   # imply that a box model exists; only building one (below) shows that.
   report_ok = helly_check(cm_ok)
   assert report_ok.complete is True
   assert not report_ok.certificates_nondegenerate  # obstruction-free

   box_model = BoxModelBuilder(cm_ok, d_max=4, time_limit=30.0).build()
   assert box_model.is_model() and box_model.is_nondegenerate()

   # Export as (lo, hi) corners or as (center, offset) for ELBE / BoxSquaredEL
   lo, hi = boxes_as_tensors(box_model, ds_ok.class_to_id)
   centers, offsets = boxes_as_tensors(
       box_model, ds_ok.class_to_id, as_center_offset=True
   )

   # Note: BoxModelBuilder minimises total box width sum(U - L), so returned boxes
   # frequently have lo == hi (offsets == 0). If seeding a gradient-based trainer
   # that requires positive box volume, clamp offsets from below before training:
   offsets_with_volume = np.maximum(offsets, 1e-3)


Initialising box embeddings from the exact model
------------------------------------------------

:class:`~mowl.geometric.init.ExactBoxInit` initialises the class boxes of
:class:`~mowl.models.BoxSquaredEL`, :class:`~mowl.models.ELBE`, and
:class:`~mowl.models.MultiBoxEL` directly from the canonical model of the
training ontology (and optionally adds an :math:`L_2^2` anchor regulariser via
``anchor_weight > 0``; defaults ``init_boxes=None, anchor_weight=0.0`` leave
behaviour unchanged):

* **Containment guarantee (projected mode,** :math:`k = 1` **):** When
  ``method="projected"`` and ``n_components=1``, every entailed atomic
  subsumption :math:`C \sqsubseteq D` with a **satisfiable** subclass :math:`C`
  satisfies :math:`\text{box}(C) \subseteq \text{box}(D)` (up to ``float32``
  tolerance ``1e-6``), so every ``gci0`` training axiom whose subclass is
  satisfiable has zero :class:`~mowl.models.BoxSquaredEL` /
  :class:`~mowl.models.ELBE` loss at margin :math:`0` immediately after
  :meth:`~mowl.geometric.init.ExactBoxInit.apply_to_module`.
* **Non-guarantees:** Axioms :math:`U \sqsubseteq A` whose subclass :math:`U` is
  **unsatisfiable** are not covered (unsatisfiable concepts and ``BOTTOM`` are
  initialised to the point box at the origin, ``center = 0, offset = 0``), and
  projected bounding boxes do **not** guarantee disjointness
  (:math:`A \sqcap B \sqsubseteq \bot`). When ``method="milp"`` is requested on
  an ontology with a non-degenerate Helly certificate (such as :math:`T_3`) or
  when the MILP search is inconclusive, construction automatically falls back to
  ``"projected"`` and records ``report["fallback"]``. Classes present in
  ``dataset.class_to_id`` but absent from the canonical model receive ``NaN`` in
  ``init.centers`` / ``init.offsets``, are left untouched by
  :meth:`~mowl.geometric.init.ExactBoxInit.apply_to_module`, and are ignored by
  :meth:`~mowl.geometric.init.ExactBoxInit.anchor_loss`. The models pass their
  ``class_index_dict`` to ``apply_to_module``, which matches rows by IRI; the
  auxiliary classes that normalisation appends to ``class_index_dict`` (IRIs
  under ``http://mowl.borg/aux/``) keep their random initialisation and are
  ignored by the anchor.

.. testcode::

   import shutil
   import tempfile
   import pathlib

   import mowl
   mowl.init_jvm("2g")  # no-op if the JVM is already running

   FIXTURES = pathlib.Path(mowl.__file__).resolve().parents[1] / "tests" / "geometric" / "fixtures"

   import torch
   from mowl.datasets import PathDataset
   from mowl.geometric import CanonicalModel, ExactBoxInit
   from mowl.models import BoxSquaredEL, ELBE

   # Work on a copy so that the EL normaliser's cache file
   # (<name>_mowl_el_normalized_v2.owl) is written next to the copy,
   # not into the source tree.
   tmpdir = tempfile.TemporaryDirectory()
   workdir = pathlib.Path(tmpdir.name)
   shutil.copy(FIXTURES / "t3.owl", workdir / "t3.owl")
   t3_dataset = PathDataset(str(workdir / "t3.owl"))
   cm = CanonicalModel.from_dataset(t3_dataset)
   init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7, seed=0)

   # 1. Containment guarantee for every entailed subsumption with satisfiable subclass
   c2i = init.class_to_id
   for a in cm.names:
       for b in cm.names:
           if cm.entails(a, b) and cm.is_satisfiable(a):
               lo_a = init.centers[c2i[a], 0] - init.offsets[c2i[a], 0]
               hi_a = init.centers[c2i[a], 0] + init.offsets[c2i[a], 0]
               lo_b = init.centers[c2i[b], 0] - init.offsets[c2i[b], 0]
               hi_b = init.centers[c2i[b], 0] + init.offsets[c2i[b], 0]
               assert (lo_b - 1e-6 <= lo_a).all() and (hi_a <= hi_b + 1e-6).all()

   # 2. Apply via init_boxes and anchor_weight hooks on BoxSquaredEL and ELBE
   b2 = BoxSquaredEL(
       t3_dataset, embed_dim=7, margin=0.0, init_boxes=init, anchor_weight=0.1
   )
   elbe = ELBE(
       t3_dataset, embed_dim=7, margin=0.0, init_boxes=init, anchor_weight=0.1
   )

   # Every gci0 training axiom (all subclasses in t3.owl are satisfiable) has zero loss at margin 0
   gci0_data = b2.training_datasets["gci0"].data
   with torch.no_grad():
       assert torch.all(b2.module(gci0_data, "gci0") <= 1e-6)

   # MILP mode falls back to "projected" when NotBoxRepresentable is raised on T3
   init_milp = ExactBoxInit.from_dataset(
       t3_dataset, embed_dim=4, method="milp", milp_time_limit=20.0
   )
   assert init_milp.method == "projected"
   assert init_milp.report["fallback"].startswith("NotBoxRepresentable")

   tmpdir.cleanup()


MultiBoxEL on :math:`T_3`
-------------------------

:class:`~mowl.models.MultiBoxEL` (:class:`~mowl.nn.MultiBoxELModule`) represents
every ontology class as a union of :math:`k = \text{n\_components}` axis-parallel
boxes in :math:`\mathbb{R}^d` under Box²EL semantics. MultiBoxEL does not
support role axioms (``role_axiom_capable = False``). Its losses have the
following properties:

* **Exact** :math:`k = 1` **identity:** When ``n_components=1`` and
  ``component_reg=0.0``, every GCI loss (positive and negative, including
  ``gci1`` batches with empty intersections) and
  :meth:`~mowl.nn.MultiBoxELModule.regularization_loss` equal
  :class:`~mowl.models.BoxSquaredEL` to within ``1e-6``.
* **Gated** ``gci1`` **and coordinatewise** ``gci1_bot``: In ``gci1``
  (:math:`C \sqcap D \sqsubseteq E`), a detached argmin over emptiness selects a
  designated component pair :math:`(i^*, j^*) = \arg\min_{i,j} \text{empt}(C_i, D_j)`,
  and the detached gate :math:`g_{ij} = 1` iff :math:`\text{lo}_{ij} \le \text{up}_{ij}`
  in every coordinate or :math:`(i, j) = (i^*, j^*)` enforces inclusion into some
  component :math:`E_m` only on intersecting pairs (plus the designated pair), so
  disjoint non-designated pairs impose nothing. In ``gci1_bot``
  (:math:`C \sqcap D \sqsubseteq \bot`), zero loss at :math:`\gamma = 0` holds
  iff :math:`\text{dist}(C_i, D_j) \ge 0` in *every* coordinate across all
  :math:`k^2` pairs (zero loss implies disjointness up to boundary contact,
  **but not conversely**).
* **Certificate triangles and** :math:`T_3`:
  :func:`~mowl.geometric.helly.certificate_triangles` enumerates non-degenerate
  singleton certificates on the graph of direct
  :class:`~mowl.geometric.normal_forms.NF2` definitions in
  :math:`O(\sum_v \deg(v)^2)` time (it is sound on all returned triangles,
  though **not complete** for conjunctions defined through chains of equivalences
  or higher-arity certificates). On ``tests/geometric/fixtures/t3.owl`` it
  returns the single triangle :math:`(X_1, X_2, X_3)` with pairwise names
  :math:`(A, B, C)`, which no non-degenerate single-box model (:math:`k = 1`) can
  satisfy, whereas :class:`~mowl.models.MultiBoxEL` with ``n_components=2`` can:

.. testcode::

   import shutil
   import tempfile
   import pathlib

   import mowl
   mowl.init_jvm("2g")  # no-op if the JVM is already running

   FIXTURES = pathlib.Path(mowl.__file__).resolve().parents[1] / "tests" / "geometric" / "fixtures"

   import numpy as np
   import torch
   from mowl.datasets import PathDataset
   from mowl.geometric import CanonicalModel, ExactBoxInit, certificate_triangles
   from mowl.models import MultiBoxEL

   NS = "http://mowl.borg/geo#"
   # Work on a copy so that the EL normaliser's cache file
   # (<name>_mowl_el_normalized_v2.owl) is written next to the copy,
   # not into the source tree.
   tmpdir = tempfile.TemporaryDirectory()
   workdir = pathlib.Path(tmpdir.name)
   shutil.copy(FIXTURES / "t3.owl", workdir / "t3.owl")
   t3_dataset = PathDataset(str(workdir / "t3.owl"))
   cm = CanonicalModel.from_dataset(t3_dataset)

   # 1. Scalable NF2 triangle finder detects the T3 Helly obstruction
   triangles = certificate_triangles(cm)
   assert len(triangles) == 1
   cert = triangles[0]
   assert (cert.c1, cert.c2, cert.c3) == ((NS + "X1",), (NS + "X2",), (NS + "X3",))
   assert set(cert.pairwise_names) == {NS + "A", NS + "B", NS + "C"}

   # 2. Train MultiBoxEL (k = 2) with k-means projected initialisation
   init_k2 = ExactBoxInit.from_dataset(
       t3_dataset, embed_dim=2, n_components=2, seed=0
   )
   model = MultiBoxEL(
       t3_dataset,
       embed_dim=2,
       n_components=2,
       margin=0.0,
       learning_rate=0.01,
       neg_sampling_gcis=[],
       init_boxes=init_k2,
   )
   model.train(epochs=3, validate_every=10)

   # Inspect the learned (B, k, d) component boxes and effective component counts
   idx = torch.tensor([model.class_index_dict[NS + "X1"]], dtype=torch.long)
   centers, offsets = model.module.class_boxes(idx)
   assert centers.shape == (1, 2, 2) and offsets.shape == (1, 2, 2)
   eff = model.module.effective_components(eps=1e-3)
   assert eff.shape == (len(model.class_index_dict),)
   assert all(torch.isfinite(p).all() for p in model.module.parameters())

   tmpdir.cleanup()
