Worked examples
===============

Canonical model (TBox and ABox: nominal encoding and instance checking)
-----------------------------------------------------------------------

:func:`~mowl.geometric.normal_forms.normalise` and
:class:`~mowl.geometric.completion.CanonicalModel` support knowledge bases with
ABox assertions by encoding each named individual :math:`a` as a fresh nominal
concept name :math:`N_a = \text{nominal\_name}(a)` (with prefix
:data:`~mowl.geometric.normal_forms.NOMINAL_PREFIX` ``"http://mowl.borg/nominal/"``
followed by the percent-encoded IRI), translating atomic class assertions
:math:`C(a)` into :class:`~mowl.geometric.normal_forms.NF1` :math:`(N_a, C)` and
object-property assertions :math:`r(a, b)` into
:class:`~mowl.geometric.normal_forms.NF3` :math:`(N_a, r, N_b)`. For
:math:`\mathcal{ELH}` (with :math:`\bot`) without nominals, functionality, or
range axioms, saturation on this encoding is sound and complete for atomic
instance checking (:meth:`~mowl.geometric.completion.CanonicalModel.instance_of`,
aliased as :meth:`~mowl.geometric.completion.CanonicalModel.entails_assertion`)
and assertion contradiction
(:meth:`~mowl.geometric.completion.CanonicalModel.assertion_contradicted`,
testing :math:`\mathcal{O} \models N_a \sqcap C \sqsubseteq \bot`).
:attr:`~mowl.geometric.completion.CanonicalModel.individuals` maps each
individual IRI to its nominal name :math:`N_a`,
:meth:`~mowl.geometric.completion.CanonicalModel.role_instances` returns the
pairs of individual IRIs connected by a role in the canonical model, and
:attr:`~mowl.geometric.completion.CanonicalModel.concept_names` returns only the
non-nominal concept names while :attr:`~mowl.geometric.completion.CanonicalModel.names`
keeps all satisfiable concept and nominal names (so constructive region builders
assign every satisfiable individual a non-empty region). If any individual is
unsatisfiable (:math:`\mathcal{O} \models N_a \sqsubseteq \bot`), constructing
:class:`~mowl.geometric.completion.CanonicalModel` raises
:class:`~mowl.geometric.errors.InconsistentOntology` with the offending
individual IRI in the message:

.. testcode::

   import mowl
   mowl.init_jvm("2g")  # no-op if the JVM is already running

   import mowl.datasets  # ensure dataset classes are loaded before completion
   from mowl.geometric import (
       BOTTOM,
       TOP,
       CanonicalModel,
       InconsistentOntology,
       individual_of_name,
       is_nominal_name,
       nominal_name,
   )
   from mowl.owlapi import OWLAPIAdapter

   NS = "http://mowl.borg/geo#"
   IND = "http://mowl.borg/geo/ind#"

   adapter = OWLAPIAdapter()
   mgr = adapter.owl_manager
   df = adapter.data_factory
   ont = adapter.create_ontology("http://mowl.borg/geo/abox_example")

   top = adapter.create_class(TOP)
   bot = adapter.create_class(BOTTOM)
   father = adapter.create_class(NS + "Father")
   mother = adapter.create_class(NS + "Mother")
   parent = adapter.create_class(NS + "Parent")
   male = adapter.create_class(NS + "Male")
   female = adapter.create_class(NS + "Female")
   has_child = adapter.create_object_property(NS + "hasChild")

   bob = adapter.create_individual(IND + "bob")
   ann = adapter.create_individual(IND + "ann")
   carl = adapter.create_individual(IND + "carl")

   # TBox: Father <= Parent, Mother <= Parent, exists hasChild.TOP <= Parent, Male & Female <= BOTTOM
   mgr.addAxiom(ont, adapter.create_subclass_of(father, parent))
   mgr.addAxiom(ont, adapter.create_subclass_of(mother, parent))
   mgr.addAxiom(
       ont,
       adapter.create_subclass_of(
           adapter.create_object_some_values_from(has_child, top), parent
       ),
   )
   mgr.addAxiom(
       ont,
       adapter.create_subclass_of(
           adapter.create_object_intersection_of(male, female), bot
       ),
   )

   # ABox: Father(bob), Male(bob), hasChild(bob, ann), hasChild(carl, ann)
   mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(father, bob))
   mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(male, bob))
   mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(has_child, bob, ann))
   mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(has_child, carl, ann))

   cm = CanonicalModel.from_ontology(ont)

   # Nominal encoding and separation of concept_names from nominals
   assert set(cm.individuals.keys()) == {IND + "bob", IND + "ann", IND + "carl"}
   bob_nom = cm.individuals[IND + "bob"]
   assert bob_nom == nominal_name(IND + "bob") and is_nominal_name(bob_nom)
   assert individual_of_name(bob_nom) == IND + "bob"
   assert bob_nom in cm.names and bob_nom not in cm.concept_names
   assert cm.element_of(IND + "bob") == cm.element_of(bob_nom)

   # Instance checking (via hierarchy and via role domain) and contradiction
   assert cm.instance_of(IND + "bob", NS + "Parent")
   assert cm.instance_of(IND + "carl", NS + "Parent")
   assert not cm.instance_of(IND + "ann", NS + "Parent")
   assert cm.assertion_contradicted(IND + "bob", NS + "Female")
   assert not cm.assertion_contradicted(IND + "bob", NS + "Parent")
   assert cm.role_instances(NS + "hasChild") == {
       (IND + "bob", IND + "ann"),
       (IND + "carl", IND + "ann"),
   }

   # Asserting Female(bob) makes bob's nominal unsatisfiable and raises InconsistentOntology
   mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(female, bob))
   try:
       CanonicalModel.from_ontology(ont)
   except InconsistentOntology as exc:
       assert (IND + "bob") in str(exc)


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
:class:`~mowl.models.BoxSquaredEL`, :class:`~mowl.models.ELBE`,
:class:`~mowl.models.MultiBoxEL`, and :class:`~mowl.models.MultiELBE` (as well
as the class prisms of :class:`~mowl.models.PrismEL`) directly from the
canonical model of the training ontology (and optionally adds an :math:`L_2^2`
anchor regulariser via ``anchor_weight > 0``; defaults
``init_boxes=None, anchor_weight=0.0`` leave behaviour unchanged):

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


MultiELBE on the Family ontology
--------------------------------

:class:`~mowl.models.MultiELBE` (:class:`~mowl.nn.MultiELBEModule`) represents
every ontology class as a union of :math:`k = \text{n\_components}` axis-parallel
boxes in :math:`\mathbb{R}^d` under ELBE inclusion, gated pairwise intersection
(``gci1``), coordinatewise separation over all :math:`k^2` pairs (``gci1_bot``),
ELBE overlap semantics for ``gci3``, translation vectors for roles, and single
boxes for individuals. At ``n_components=1`` with
``component_aggregation="hard"``, ``component_reg=0.0``, and
``diversity_reg=0.0``, :class:`~mowl.nn.MultiELBEModule` reproduces
:class:`~mowl.nn.ELBEModule` on every GCI loss to within ``1e-6`` at equal
``margin`` (note that :class:`~mowl.models.MultiELBE` defaults to ``margin=0.1``
like :class:`~mowl.nn.ELBEModule`, whereas :class:`~mowl.models.ELBE` defaults
to ``margin=0``), and accepts ``init_boxes`` and ``anchor_weight`` from
:class:`~mowl.geometric.init.ExactBoxInit` via the ``class_center`` alias for
``class_embed``:

.. testcode::

   import mowl
   mowl.init_jvm("2g")  # no-op if the JVM is already running

   import torch
   from mowl.datasets.builtin import FamilyDataset
   from mowl.geometric import ExactBoxInit
   from mowl.models import ELBE, MultiELBE

   ds = FamilyDataset()

   # 1. Exact k = 1 identity with ELBE at equal margin
   elbe = ELBE(ds, embed_dim=6, margin=0.1)
   melbe_k1 = MultiELBE(ds, embed_dim=6, margin=0.1, n_components=1)
   melbe_k1.module.load_state_dict(elbe.module.state_dict(), strict=False)
   gci0_batch = elbe.training_datasets["gci0"].data
   with torch.no_grad():
       diff = (
           elbe.module(gci0_batch, "gci0")
           - melbe_k1.module(gci0_batch, "gci0")
       ).abs().max()
       assert float(diff.item()) <= 1e-6

   # 2. Multi-component (k = 2) training with ExactBoxInit and anchor_weight
   init_k2 = ExactBoxInit.from_dataset(ds, embed_dim=6, n_components=2, seed=0)
   model = MultiELBE(
       ds,
       embed_dim=6,
       margin=0.1,
       n_components=2,
       init_boxes=init_k2,
       anchor_weight=0.1,
   )
   model.train(epochs=3, validate_every=10)
   eff = model.module.effective_components(eps=1e-3)
   assert eff.shape == (len(model.class_index_dict),)
   assert all(torch.isfinite(p).all() for p in model.module.parameters())


PrismEL on the Family ontology
------------------------------

:class:`~mowl.models.PrismEL` (:class:`~mowl.nn.PrismModule`) represents every
ontology class :math:`C` as a prism
:math:`P_C = T_C \times B_C \subset \mathbb{R}^2 \times \mathbb{R}^{d-2}` with
:math:`2d - 1` scalar parameters per class (triangle centre ``(2)``, triangle
inradius ``(1)``, box centre ``(d - 2)``, and box half-width ``(d - 2)``,
reported by :meth:`~mowl.models.PrismEL.count_class_parameters`) following
Federico, Leemhuis & Kutz (DL 2026). When initialised with
:class:`~mowl.geometric.init.ExactBoxInit`, coordinates ``2..d`` of each
initialisation box populate ``(box_center, box_offset)`` and coordinates
``0..1`` populate the equilateral triangle with ``tri_center = center[:2]`` and
:math:`\text{tri\_inradius} = \max(o_0, o_1) \sqrt{3}` (which suffices to
contain the rectangle in coordinates ``0..1``, not minimal), so that
every corner of the initialisation box lies inside the initialised prism:

.. testcode::

   import mowl
   mowl.init_jvm("2g")  # no-op if the JVM is already running

   import torch
   from mowl.datasets.builtin import FamilyDataset
   from mowl.geometric import ExactBoxInit
   from mowl.models import PrismEL

   ds = FamilyDataset()
   init = ExactBoxInit.from_dataset(ds, embed_dim=6, seed=0)
   model = PrismEL(
       ds,
       embed_dim=6,
       margin=0.1,
       init_boxes=init,
       anchor_weight=0.1,
   )
   assert model.count_class_parameters() == 2 * 6 - 1 == 11
   model.train(epochs=3, validate_every=10)
   assert all(torch.isfinite(p).all() for p in model.module.parameters())


CanonicalMixture on a tiny ABox knowledge base
----------------------------------------------

:class:`~mowl.models.CanonicalMixture` (:class:`~mowl.nn.CanonicalMixtureModule`)
represents each individual :math:`a` as a probability distribution
:math:`p_a = \operatorname{softmax}_{S_a}(\ell_a)` over its admissible canonical
support
:math:`S_a = \{x \in S : x \in C^{\mathcal{I}} \text{ for every training-entailed class } C \text{ of } a\}`,
where :math:`S` and the sparse membership matrix
:attr:`~mowl.geometric.membership.MembershipStructure.M` come from the
:class:`~mowl.geometric.completion.CanonicalModel` of the training ontology. By
construction, every training-entailed class assertion (including assertions
entailed through role assertions) has empty complement in :math:`S_a` and scores
``0.0`` exactly, while every contradicted class assertion has zero probability
on :math:`S_a` and scores ``+inf``:

.. testcode::

   import math
   import pathlib
   import tempfile

   import mowl
   mowl.init_jvm("2g")  # no-op if the JVM is already running

   import torch
   from org.semanticweb.owlapi.model import IRI
   from mowl.datasets import PathDataset
   from mowl.geometric.normal_forms import BOTTOM, TOP
   from mowl.models import CanonicalMixture
   from mowl.owlapi import OWLAPIAdapter

   NS = "http://mowl.borg/geo/ex#"
   IND = "http://mowl.borg/geo/ex/ind#"

   adapter = OWLAPIAdapter()
   mgr = adapter.owl_manager
   df = adapter.data_factory
   ont = adapter.create_ontology("http://mowl.borg/geo/ex/ont")

   top = adapter.create_class(TOP)
   bot = adapter.create_class(BOTTOM)
   father = adapter.create_class(NS + "Father")
   parent = adapter.create_class(NS + "Parent")
   male = adapter.create_class(NS + "Male")
   female = adapter.create_class(NS + "Female")
   has_child = adapter.create_object_property(NS + "hasChild")
   bob = adapter.create_individual(IND + "bob")
   ann = adapter.create_individual(IND + "ann")

   # TBox: Father <= Parent, Father <= Male, exists hasChild.TOP <= Parent, Male & Female <= BOTTOM
   mgr.addAxiom(ont, adapter.create_subclass_of(father, parent))
   mgr.addAxiom(ont, adapter.create_subclass_of(father, male))
   mgr.addAxiom(
       ont,
       adapter.create_subclass_of(
           adapter.create_object_some_values_from(has_child, top), parent
       ),
   )
   mgr.addAxiom(
       ont,
       adapter.create_subclass_of(
           adapter.create_object_intersection_of(male, female), bot
       ),
   )

   # ABox: Father(bob), Female(ann), hasChild(bob, ann)
   mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(father, bob))
   mgr.addAxiom(ont, df.getOWLClassAssertionAxiom(female, ann))
   mgr.addAxiom(ont, df.getOWLObjectPropertyAssertionAxiom(has_child, bob, ann))

   tmpdir = tempfile.TemporaryDirectory()
   owl_path = pathlib.Path(tmpdir.name) / "tiny_abox.owl"
   mgr.saveOntology(ont, IRI.create(owl_path.as_uri()))

   ds = PathDataset(str(owl_path))
   model = CanonicalMixture(
       ds,
       rank=8,
       lr=0.01,
       batch_size=2,
       model_filepath=str(pathlib.Path(tmpdir.name) / "cm.pt"),
   )
   model.train(epochs=3, validate_every=1)

   bob_idx = model.individual_index_dict[IND + "bob"]
   parent_idx = model.class_index_dict[NS + "Parent"]
   female_idx = model.class_index_dict[NS + "Female"]

   entailed_score = model.module.class_assertion_score(
       torch.tensor([[bob_idx, parent_idx]], dtype=torch.long)
   )
   contradicted_score = model.module.class_assertion_score(
       torch.tensor([[bob_idx, female_idx]], dtype=torch.long)
   )
   assert float(entailed_score.item()) == 0.0
   assert math.isinf(float(contradicted_score.item())) and float(contradicted_score.item()) > 0.0

   tmpdir.cleanup()


Objective O1: Soft component aggregation and diversity regularisation
---------------------------------------------------------------------

Under hard aggregation (``component_aggregation="hard"``), taking ``amin`` over
target components and ``argmin`` over pairwise emptiness routes gradient only to
the single nearest component, leaving non-nearest components dead from random
initialisation. Objective O1 (implemented in :mod:`mowl.nn.el.softagg` and
available on both :class:`~mowl.models.MultiBoxEL` and
:class:`~mowl.models.MultiELBE`) sets ``component_aggregation="soft"`` to
replace every target minimum by the normalised log-mean-exp soft-minimum
:func:`~mowl.nn.el.softagg.softmin`
(:math:`\text{softmin}_\tau(v) = -\tau(\log\sum_{j=1}^K \exp(-v_j/\tau) - \log K)`)
and ``gci1``'s designated pair by :func:`~mowl.nn.el.softagg.soft_gci1_terms`
(softmax emptiness weights :math:`w_{ij} = \text{softmax}(-\text{empt}_{ij}/\tau)`
with a detached selection gate
:math:`g_{ij} = \max(\text{non\_empty}_{ij}^{\text{det}}, w_{ij}^{\text{det}})`).
When ``tau_anneal=True``, ``module.tau`` (stored as a registered buffer in
``state_dict()``) is annealed geometrically from ``tau_start`` (default ``1.0``)
to ``tau_end`` (default ``0.05``) across the epochs of each ``train()`` call;
``tau_end`` defaults to ``0.05`` rather than ``0.01`` because ``float32``
softmax weights :math:`\exp(-\Delta/\tau)` underflow to exact ``0.0`` once the
score gap :math:`\Delta` exceeds :math:`\approx 100\tau`. Because
:math:`\text{softmin}_\tau(v) \le \text{mean}(v)` pulls all target components
together at finite :math:`\tau > 0`, ``diversity_reg`` adds the log-space
pairwise soft containment ratio :func:`~mowl.nn.el.softagg.diversity_penalty`,
which evaluates to ``1.0`` on identical boxes at any ambient dimension :math:`d`
and to :math:`\le 10^{-3}` when components are disjoint:

.. testcode::

   import mowl
   mowl.init_jvm("2g")  # no-op if the JVM is already running

   import torch
   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import MultiBoxEL, MultiELBE
   from mowl.nn.el.softagg import diversity_penalty, softmin

   ds = FamilyDataset()

   # 1. Soft aggregation with geometric temperature annealing and diversity_reg
   mb = MultiBoxEL(
       ds,
       embed_dim=8,
       n_components=2,
       component_aggregation="soft",
       tau_start=1.0,
       tau_end=0.05,
       tau_anneal=True,
       diversity_reg=0.1,
       neg_sampling_gcis=[],
   )
   mb.train(epochs=3, validate_every=10)
   assert abs(float(mb.module.tau.item()) - 0.05) < 1e-6
   assert "tau" in mb.module.state_dict()

   me = MultiELBE(
       ds,
       embed_dim=8,
       n_components=2,
       component_aggregation="soft",
       tau_start=1.0,
       tau_end=0.05,
       tau_anneal=True,
       diversity_reg=0.1,
       neg_sampling_gcis=[],
   )
   me.train(epochs=3, validate_every=10)
   assert abs(float(me.module.tau.item()) - 0.05) < 1e-6

   # 2. Shared helpers in mowl.nn.el.softagg: softmin and log-space diversity_penalty
   v = torch.tensor([[0.0, 2.0]])
   assert float(softmin(v, tau=1e-4, dim=-1).item()) < 1e-3
   ident_c = torch.zeros((1, 2, 50), requires_grad=True)
   ident_o = torch.full((1, 2, 50), 0.2)
   pen = diversity_penalty(ident_c, ident_o)
   pen.backward()
   assert abs(float(pen.item()) - 1.0) < 1e-5
   assert float(ident_c.grad.norm().item()) > 0.0


Objective O3: Canonical-element point supervision
-------------------------------------------------

Objective O3 supervises class regions by points—the canonical elements of
satisfiable named classes and ABox individuals in the training ontology's
:class:`~mowl.geometric.completion.CanonicalModel`—rather than by box-to-box
inclusion. :class:`~mowl.datasets.el.points.PointSupervision` (in
:mod:`mowl.datasets.el.points`) precomputes positive membership pairs
:math:`(x_D, C)` where :math:`\mathcal{O} \models D \sqsubseteq C`, hard
negative pairs where :math:`\mathcal{O} \models D \sqcap C \sqsubseteq \bot`,
per-element pools of non-entailed classes for random negatives (weighted by
``neg_weight``, default ``0.1``), and canonical role edges :math:`(x_C, r, x_D)`,
sampling up to ``points_per_epoch`` (default ``100_000``) pairs per epoch. In
:mod:`mowl.nn.el.pointloss`, soft coordinate-minimum box membership
:math:`m(x; c, o) = \min_{\text{coord}} \sigma((|o| - |x - c|)/\beta)` (with
temperature ``beta``, default ``0.1``) is combined across components via
noisy-OR :math:`m_C(x) = 1 - \prod_i (1 - m_{C_i}(x))` (evaluating individual
targets in role edges against a virtual box of half-width ``ind_halfwidth``,
defaulting to :math:`5\beta`, around ``ind_embed``) and trained with binary
cross-entropy plus the mean log-side-length regulariser ``volume_reg`` (default
``1e-3``) so boxes shrink tightly onto their points. Setting
``objective="point"`` uses only the point and volume losses (zeroing all GCI
losses), ``objective="mixed"`` keeps the backbone's :math:`\bot`-axiom
region losses (``gci0_bot``, ``gci1_bot``, ``gci3_bot``) so empty-space
disjointness is still enforced, and ``objective="hybrid"`` keeps all GCI
losses alongside the point losses. Under ``"point"``, ``"mixed"``, and
``"hybrid"``, ABox class assertions are scored by :math:`-\log m_C(a)`
(:meth:`~mowl.nn.MultiBoxELModule.class_assertion_score_point`, aliasing
individual elements directly to ``ind_embed`` without using ``ind_offset``) and
point satisfaction is reported by
:meth:`~mowl.nn.MultiBoxELModule.point_modelness` (fractions of positive pairs
with :math:`m \ge 0.5` and hard negatives with :math:`m < 0.5`):

.. testcode::

   import mowl
   mowl.init_jvm("2g")  # no-op if the JVM is already running

   import torch
   from mowl.datasets.builtin import FamilyDataset
   from mowl.datasets.el.points import PointSupervision
   from mowl.geometric import CanonicalModel
   from mowl.models import MultiBoxEL, MultiELBE

   ds = FamilyDataset()

   # Train MultiELBE with objective="mixed" (point supervision + bottom-axiom region losses)
   melbe = MultiELBE(
       ds,
       embed_dim=8,
       n_components=2,
       objective="mixed",
       beta=0.1,
       neg_weight=0.1,
       volume_reg=1e-3,
       points_per_epoch=500,
   )
   cm = CanonicalModel.from_dataset(ds)
   ps = PointSupervision.from_canonical_model(
       cm,
       melbe.named_class_index_dict,
       melbe.individual_index_dict,
       melbe.object_property_index_dict,
   )
   assert ps.n_elements > 0 and ps.pos_pairs.shape[0] > 0 and ps.hard_neg_pairs.shape[0] > 0

   melbe.train(epochs=3, validate_every=10)
   stats = melbe.module.point_modelness()
   assert 0.0 <= stats["pos_frac"] <= 1.0 and 0.0 <= stats["hardneg_frac"] <= 1.0

   # Under "mixed", gci0_loss is zeroed while gci1_bot_loss remains active
   gci0_data = melbe.training_datasets["gci0"].data
   gci1_bot_data = melbe.training_datasets["gci1_bot"].data
   with torch.no_grad():
       assert float(melbe.module(gci0_data, "gci0").abs().max().item()) == 0.0
       assert float(melbe.module(gci1_bot_data, "gci1_bot").abs().max().item()) > 0.0

   # Train MultiBoxEL with objective="point"
   mbox = MultiBoxEL(
       ds,
       embed_dim=8,
       n_components=2,
       objective="point",
       beta=0.1,
       neg_weight=0.1,
       volume_reg=1e-3,
       points_per_epoch=500,
   )
   mbox.train(epochs=3, validate_every=10)
   assert set(mbox.module.point_modelness().keys()) == {"pos_frac", "hardneg_frac"}
