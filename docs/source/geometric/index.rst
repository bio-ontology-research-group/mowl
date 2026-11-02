Constructive geometric models of EL++ ontologies
=================================================

.. |canonical| replace:: :class:`CanonicalModel <mowl.geometric.completion.CanonicalModel>`
.. |polyhedral| replace:: :class:`ConvexPolyhedralModel <mowl.geometric.polyhedral.ConvexPolyhedralModel>`
.. |intervals| replace:: :class:`IntervalUnionModel <mowl.geometric.intervals.IntervalUnionModel>`
.. |boxes| replace:: :class:`BoxModelBuilder <mowl.geometric.boxes.BoxModelBuilder>`
.. |boxmodel| replace:: :class:`BoxModel <mowl.geometric.boxes.BoxModel>`
.. |checker| replace:: :class:`ModelChecker <mowl.geometric.checker.ModelChecker>`
.. |helly| replace:: :func:`helly_check <mowl.geometric.helly.helly_check>`

Learned geometric ontology embeddings in mOWL—such as
:class:`ELEmbeddings <mowl.models.elembeddings.model.ELEmbeddings>`,
:class:`ELBE <mowl.models.elbe.model.ELBE>`, and
:class:`BoxSquaredELModule <mowl.nn.el.boxsquaredel.module.BoxSquaredELModule>`—optimise
differentiable loss functions and satisfy an ontology only approximately.
Recent theoretical results characterise when *exact* geometric region models of
:math:`\mathcal{EL}^{++}` ontologies exist and give explicit constructions:

* **Axis-parallel boxes** (and Euclidean balls) cannot model the six-axiom
  propositional theory

  .. math::

     T_3 = \{A \equiv X_1 \sqcap X_2,\; B \equiv X_2 \sqcap X_3,\; C \equiv X_1 \sqcap X_3,\;
             A \sqcap B \sqsubseteq \bot,\; B \sqcap C \sqsubseteq \bot,\; A \sqcap C \sqsubseteq \bot\}

  in any non-degenerate interpretation in any dimension :math:`d \ge 1`, because
  axis-parallel boxes in :math:`\mathbb{R}^d` have Helly number 2 (Eckhoff, 1988;
  Leemhuis & Kutz, NeSy 2025).
* **Convex polyhedra** model every consistent normalised :math:`\mathcal{EL}^{++}`
  ontology with a finite canonical model *exactly* in ambient dimension
  :math:`k = |\Delta^{\mathcal{I}}| - 1` by mapping each concept name to a face
  of the standard simplex and each role to the convex hull of concatenated vertex
  pairs (Gutiérrez-Basulto & Schockaert, KR 2018, Prop. 3; Lacerda, Ozaki &
  Guimarães, TGDK 2024, §5). The dimension :math:`|\Delta^{\mathcal{I}}| - 1`
  includes auxiliary names and :math:`\top` and is an upper bound; a dimension
  linear in the number of concept names is unavoidable in the worst case for
  convex models that realise the relevant conjunctions (Gutiérrez-Basulto &
  Schockaert, KR 2018, Prop. 4).
* **Unions of closed intervals on the real line** model every consistent
  normalised :math:`\mathcal{EL}^{++}` ontology *exactly* in one dimension, with
  roles interpreted as finite unions of 2-D axis-aligned rectangles (Lacerda,
  Ozaki & Guimarães, TGDK 2024, Def. 12; the :math:`n = 1` case of the multibox
  construction of Federico, Leemhuis & Kutz, JOWO 2026, Thm. 3).

The :mod:`mowl.geometric` package and its learnable counterparts follow a
five-part progression:

1. **Canonical model (TBox, RBox, and ABox):** :func:`~mowl.geometric.normal_forms.normalise`
   and :class:`~mowl.geometric.completion.CanonicalModel` saturate normalised
   :math:`\mathcal{EL}^{++}` TBoxes, RBoxes, and ABoxes (encoding each named
   individual :math:`a` as a fresh nominal concept name :math:`N_a` via
   :func:`~mowl.geometric.normal_forms.nominal_name`, atomic class assertions
   :math:`C(a)` as :math:`N_a \sqsubseteq C`, and role assertions :math:`r(a, b)`
   as :math:`N_a \sqsubseteq \exists r.N_b`, while raising
   :class:`~mowl.geometric.errors.InconsistentOntology` if any individual is
   unsatisfiable).
2. **Exact constructive models and diagnostics:**
   :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`,
   :class:`~mowl.geometric.intervals.IntervalUnionModel`,
   :class:`~mowl.geometric.boxes.BoxModelBuilder`,
   :class:`~mowl.geometric.checker.ModelChecker`,
   :func:`~mowl.geometric.helly.helly_check`, and
   :func:`~mowl.geometric.helly.certificate_triangles`.
3. **Exact-model box initialisation and anchoring:**
   :class:`~mowl.geometric.init.ExactBoxInit` seeds and anchors
   :class:`~mowl.models.BoxSquaredEL`, :class:`~mowl.models.ELBE`,
   :class:`~mowl.models.MultiBoxEL`, :class:`~mowl.models.MultiELBE`, and
   :class:`~mowl.models.PrismEL`.
4. **Learnable geometric and canonical-mixture models:** ``init_boxes`` and
   ``anchor_weight`` hooks on :class:`~mowl.models.ELBE` and
   :class:`~mowl.models.BoxSquaredEL`, multi-component box models
   :class:`~mowl.models.MultiBoxEL` and :class:`~mowl.models.MultiELBE`, prism
   embeddings :class:`~mowl.models.PrismEL`, and the faithful-by-construction
   canonical-element mixture model :class:`~mowl.models.CanonicalMixture`
   (backed by :class:`~mowl.geometric.membership.MembershipStructure`).
5. **Training objectives for multi-box and prism models:** soft component
   aggregation and log-space volume-overlap diversity regularisation in
   :mod:`mowl.nn.el.softagg`, canonical-element point
   supervision in :mod:`mowl.datasets.el.points` and
   :mod:`mowl.nn.el.pointloss`, available via
   ``objective="point" | "mixed" | "hybrid"`` on :class:`~mowl.models.MultiBoxEL` and
   :class:`~mowl.models.MultiELBE`, and structural region algebra, compiled
   conjunction definition tables, and structural losses in
   :mod:`mowl.nn.el.structural`, available via ``objective="structural"`` on
   :class:`~mowl.models.MultiBoxEL` and :class:`~mowl.models.PrismEL`.


Guarantees by model and procedure
---------------------------------

Every constructed model (:class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`,
:class:`~mowl.geometric.intervals.IntervalUnionModel`, and
:class:`~mowl.geometric.boxes.BoxModel`) is verified by
:class:`~mowl.geometric.checker.ModelChecker` before it is returned; if a
constructed model ever fails verification, construction raises
:class:`~mowl.geometric.errors.InternalConsistencyError`.

.. list-table:: Constructive models, learnable models, and objectives in ``mowl.geometric``
   :header-rows: 1
   :widths: 22 20 38 20

   * - Class / function
     - Supported fragment
     - Mathematical guarantee
     - Dimension
   * - :class:`~mowl.geometric.completion.CanonicalModel`
     - Normalised :math:`\mathcal{EL}^{++}` TBox/RBox (:class:`~mowl.geometric.normal_forms.NF1`–:class:`~mowl.geometric.normal_forms.NF4`, :class:`~mowl.geometric.normal_forms.NF6`, :class:`~mowl.geometric.normal_forms.NF7`, and :math:`\bot` forms) and atomic ABox assertions
     - Sound and complete on concept-name subsumptions and unsatisfiability wrt the normalised axiom set (`Kazakov et al., JAR 2014`), and sound and complete for atomic instance checking (:meth:`~mowl.geometric.completion.CanonicalModel.instance_of`) and assertion contradiction (:meth:`~mowl.geometric.completion.CanonicalModel.assertion_contradicted`) in :math:`\mathcal{ELH}_\bot` without nominals, functionality, or range axioms via the nominal encoding :math:`N_a`. Raises :class:`~mowl.geometric.errors.InconsistentOntology` iff :math:`\mathcal{O} \models \top \sqsubseteq \bot` or :math:`\mathcal{O} \models N_a \sqsubseteq \bot` for some individual :math:`a`.
     - :math:`n = |\Delta^{\mathcal{I}}|` canonical elements (satisfiable concept and nominal names in :math:`N_C \cup \{N_a\} \cup \{\top\}`)
   * - :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`
     - Full normalised :math:`\mathcal{EL}^{++}` (or propositional-only when ``thicken=True``)
     - Exact model (:math:`\eta \models \mathcal{O}`), non-degenerate (:math:`\eta(A) \neq \emptyset` for all satisfiable :math:`A`), and strongly faithful on concept names (:math:`\eta(A) \subseteq \eta(B) \iff \mathcal{O} \models A \sqsubseteq B`) (`Gutiérrez-Basulto & Schockaert, KR 2018, Prop. 3`).
     - :math:`k = n - 1`, an upper bound that includes auxiliary names and :math:`\top`; a dimension linear in the number of concept names is unavoidable in the worst case for convex models that realise the relevant conjunctions (`Gutiérrez-Basulto & Schockaert, KR 2018, Prop. 4`)
   * - :class:`~mowl.geometric.intervals.IntervalUnionModel`
     - Full normalised :math:`\mathcal{EL}^{++}`
     - Exact model (:math:`\eta \models \mathcal{O}`), non-degenerate, and strongly faithful on concept names (`Lacerda et al., TGDK 2024, Def. 12`; `Federico et al., JOWO 2026, Thm. 3`).
     - :math:`1` (classes in :math:`\mathbb{R}`, roles in :math:`\mathbb{R}^2`)
   * - :func:`~mowl.geometric.helly.helly_check`
     - Propositional :math:`\mathcal{EL}_\bot` (exact for strong reading when ``complete=True``); sound necessary condition with roles
     - Every :class:`~mowl.geometric.helly.HellyCertificate` is a sound proof that no box model realises its three pairwise conjunctions simultaneously (`Eckhoff, 1988`; `Leemhuis & Kutz, NeSy 2025, Props. 4, 5`). Non-degenerate certificates exclude every non-degenerate box model; strong certificates exclude *conjunction-faithful* box models (every satisfiable conjunction of names non-empty), not box models that are merely strongly faithful on names. ``box_representable_* = True`` means "no Helly obstruction of this kind was found", not that a box model exists. Re-saturates per candidate set; intended for small to medium ontologies.
     - Dimension-independent (excludes box models in *any* :math:`\mathbb{R}^d`)
   * - :func:`~mowl.geometric.helly.certificate_triangles`
     - Any normalised :math:`\mathcal{EL}^{++}` canonical model (graph of direct :class:`~mowl.geometric.normal_forms.NF2` definitions)
     - Sound non-degenerate singleton certificates :math:`(X_1, X_2, X_3)` with :math:`X_1 < X_2 < X_3` lexicographically and all three ``pairwise_names`` non-``None`` and satisfiable. Returns every singleton non-degenerate certificate whose pairwise conjunctions are direct ``NF2`` definitions in :math:`O(\sum_v \deg(v)^2)` triangle-enumeration time; **not complete** for conjunctions defined through chains of equivalences or higher-arity certificates (:math:`|C_i| > 1`).
     - Dimension-independent (excludes non-degenerate box models in *any* :math:`\mathbb{R}^d`)
   * - :class:`~mowl.geometric.boxes.BoxModelBuilder`
     - Propositional :math:`\mathcal{EL}_\bot` only (:class:`~mowl.geometric.normal_forms.NF1`, :class:`~mowl.geometric.normal_forms.NF2`, and :math:`\bot` forms)
     - Runs :func:`~mowl.geometric.helly.helly_check` first and, in both ``faithful`` modes, raises :class:`~mowl.geometric.errors.NotBoxRepresentable` iff a non-degenerate certificate is found (a proof that no non-degenerate box model exists); searches via MILP (`scipy.optimize.milp`) and verifies any found :class:`~mowl.geometric.boxes.BoxModel` with :class:`~mowl.geometric.checker.ModelChecker` (a time-limited incumbent is post-checked and returned only if it passes). Raises :class:`~mowl.geometric.errors.BoxModelNotFound` when inconclusive.
     - Searches :math:`d \in \{1, 2, 4, \dots, d_{\max}\}` (default :math:`d_{\max} = 8`)
   * - :class:`~mowl.geometric.checker.ModelChecker`
     - ``"interval"``, ``"polyhedral"`` (full normalised :math:`\mathcal{EL}^{++}`), and ``"box"`` (propositional :math:`\mathcal{EL}_\bot`)
     - Exact decision procedure for axiom satisfaction, non-degeneracy, and strong faithfulness on concept names (combinatorial for intervals/boxes/simplex faces; HiGHS linear programming for general polyhedra and polyhedral role axioms). Polyhedral role axioms are checked only for simplex-vertex regions (Face rows and PairHull halves one-hot or zero); other regions raise :class:`~mowl.geometric.errors.UnsupportedRegion`.
     - Matches interpretation ambient dimension
   * - :class:`~mowl.geometric.init.ExactBoxInit`
     - Normalised :math:`\mathcal{EL}^{++}` (``method="projected"``) or propositional :math:`\mathcal{EL}_\bot` with fallback to ``"projected"`` (``method="milp"``)
     - In ``"projected"`` mode with :math:`k = 1`, guarantees :math:`\text{box}(C) \subseteq \text{box}(D)` (up to ``float32`` tolerance ``1e-6``) and zero ``gci0`` loss at margin :math:`0` for every entailed atomic subsumption :math:`C \sqsubseteq D` with **satisfiable** :math:`C` (unsatisfiable subclasses are points at the origin and are not covered; disjointness is not guaranteed). In ``"milp"`` mode, preserves both containment and disjointness when a box model is found (its boxes are mapped coordinate-wise by an increasing affine map from the solver's universe :math:`[0, M]` onto :math:`[-\text{scale}, \text{scale}]`), and falls back to ``"projected"`` on :class:`~mowl.geometric.errors.NotBoxRepresentable` or :class:`~mowl.geometric.errors.BoxModelNotFound`. For :class:`~mowl.nn.PrismModule`, sets the equilateral triangle in coordinates ``0..1`` to inradius :math:`\max(o_0, o_1)\sqrt{3}`, which suffices to contain the rectangle (not minimal), so the prism contains the initialisation box. Skips classes absent from the canonical model (``NaN`` rows left untouched by :meth:`~mowl.geometric.init.ExactBoxInit.apply_to_module` and ignored by :meth:`~mowl.geometric.init.ExactBoxInit.anchor_loss`).
     - Target :math:`(k, d)` matching :class:`~mowl.models.BoxSquaredEL`, :class:`~mowl.models.ELBE`, :class:`~mowl.models.PrismEL` (:math:`k = 1, d \ge 3`), :class:`~mowl.models.MultiBoxEL`, or :class:`~mowl.models.MultiELBE` (:math:`k \ge 1`)
   * - :class:`~mowl.models.MultiBoxEL` / :class:`~mowl.nn.MultiBoxELModule`
     - :math:`\mathcal{EL}^{++}` concept normal forms (``gci0``, ``gci1``, ``gci2``, ``gci3``, ``gci0_bot``, ``gci1_bot``, ``gci3_bot``) and ABox assertions (``role_axiom_capable = False``)
     - Every class is a union of :math:`k = \text{n\_components}` axis-parallel boxes with shared class bump and Box²EL role semantics (`Federico, Leemhuis & Kutz, JOWO 2026`). At :math:`k = 1` with ``component_aggregation="hard"``, ``component_reg=0.0``, and ``diversity_reg=0.0``, every GCI loss (positive and negative, including ``gci1`` batches with empty intersections) and regularisation loss equal :class:`~mowl.models.BoxSquaredEL` to within ``1e-6``. Supports soft component aggregation (``component_aggregation="soft"``, ``tau_anneal``, ``diversity_reg``) and canonical-element point supervision (``objective="point" | "mixed" | "hybrid"``).
     - :math:`k` boxes in :math:`\mathbb{R}^d` per class (:math:`k \ge 2` escapes the Helly number 2 obstruction on :math:`T_3`)
   * - :class:`~mowl.models.MultiELBE` / :class:`~mowl.nn.MultiELBEModule`
     - :math:`\mathcal{EL}^{++}` concept normal forms (``gci0``, ``gci1``, ``gci2``, ``gci3``, ``gci0_bot``, ``gci1_bot``, ``gci3_bot``) and ABox assertions (``role_axiom_capable = False``)
     - Every class is a union of :math:`k = \text{n\_components}` axis-parallel boxes with ELBE inclusion, gated pairwise intersection, translation roles, and box individuals (`Federico, Leemhuis & Kutz, JOWO 2026`). At :math:`k = 1` with ``component_aggregation="hard"``, ``component_reg=0.0``, and ``diversity_reg=0.0``, every GCI loss and MSE training loss equal :class:`~mowl.models.ELBE` / :class:`~mowl.nn.ELBEModule` at equal ``margin`` to within ``1e-6`` (preserving ELBE's overlap formula for ``gci3``). Supports ``init_boxes``/``anchor_weight``, soft component aggregation, and canonical-element point supervision.
     - :math:`k` boxes in :math:`\mathbb{R}^d` per class
   * - :class:`~mowl.models.PrismEL` / :class:`~mowl.nn.PrismModule`
     - :math:`\mathcal{EL}^{++}` concept normal forms and ABox assertions (``role_axiom_capable = False``)
     - Every class is a prism :math:`P_C = T_C \times B_C \subset \mathbb{R}^2 \times \mathbb{R}^{d-2}` (an apex-up equilateral triangle in support coordinates times an axis-parallel box) with closed-form intersection in ``gci1`` and translational roles (`Federico, Leemhuis & Kutz, DL 2026`).
     - :math:`2d - 1` scalar parameters per class in :math:`\mathbb{R}^d` (:math:`d \ge 3`)
   * - :class:`~mowl.models.CanonicalMixture` / :class:`~mowl.nn.CanonicalMixtureModule`
     - Normalised :math:`\mathcal{EL}^{++}` knowledge bases with individuals (ABox completion)
     - Represents each individual :math:`a` as a masked softmax distribution :math:`p_a` over its admissible canonical support :math:`S_a \subseteq S` from :class:`~mowl.geometric.membership.MembershipStructure`. Guarantees hierarchy violation rate ``0``, exact score ``0.0`` on all training-entailed class assertions, and ``+inf`` score on all contradicted class assertions for all parameter values.
     - Low-rank factorisation :math:`r = \text{rank}` over :math:`|S|` canonical elements
   * - Soft component aggregation (:mod:`mowl.nn.el.softagg`)
     - :class:`~mowl.models.MultiBoxEL` and :class:`~mowl.models.MultiELBE` (``component_aggregation="soft"``)
     - Replaces hard :math:`\min` over target components by :math:`\text{softmin}_\tau(v) = -\tau(\log\sum_j e^{-v_j/\tau} - \log K)` (converging to :math:`\min` as :math:`\tau \to 0^+` and equal to the single slice at :math:`K = 1`), soft-argmin emptiness weights :math:`w_{ij} = \text{softmax}(-\text{empt}_{ij}/\tau)` with detached gate :math:`g_{ij} = \max(\text{non\_empty}_{ij}^{\text{det}}, w_{ij}^{\text{det}})` in ``gci1``, and dimension-independent log-space :func:`~mowl.nn.el.softagg.diversity_penalty` (``1.0`` on identical boxes at any :math:`d`).
     - Temperature :math:`\tau` annealed geometrically from ``tau_start`` (``1.0``) to ``tau_end`` (``0.05``)
   * - Canonical-element point supervision (:class:`~mowl.datasets.el.points.PointSupervision`, :mod:`mowl.nn.el.pointloss`)
     - :class:`~mowl.models.MultiBoxEL` and :class:`~mowl.models.MultiELBE` (``objective="point" | "mixed" | "hybrid"``)
     - Supervises class regions via soft coordinate-minimum box membership :math:`m(x) = \min_{\text{coord}} \sigma((|o| - |x - c|)/\beta)` and noisy-OR union membership :math:`m_C(x) = 1 - \prod_i (1 - m_{C_i}(x))` over canonical elements of satisfiable named classes and individuals (aliasing individual elements to ``ind_embed``, with a virtual box of half-width ``ind_halfwidth`` defaulting to :math:`5\beta` around ``ind_embed`` for individual targets in role edges), plus ``volume_reg`` log-side-length regularisation. Under ``"point"`` and ``"mixed"``, ABox ``class_assertion`` is still trained in the main loop via :math:`-\log m_C(a)` while ``object_property_assertion`` is zeroed; ``"mixed"`` retains the backbone's :math:`\bot`-axiom losses (``gci0_bot``, ``gci1_bot``, ``gci3_bot``), and ``"hybrid"`` retains all GCI and ABox losses alongside the point losses. Class assertions are scored by :math:`-\log m_C(a)`.
     - :math:`|\text{elements}| \times d` point table ``elem_embed``
   * - Structural objective (:mod:`mowl.nn.el.structural`)
     - :class:`~mowl.models.MultiBoxEL` and :class:`~mowl.models.PrismEL` (``objective="structural"``)
     - Materialises regions with signed half-widths (:class:`~mowl.nn.el.structural.Regions` and :class:`~mowl.nn.el.prism.losses.PrismRegions`), compiles acyclic conjunction definitions :math:`A \equiv X \sqcap Y` (:class:`~mowl.nn.el.structural.DefinitionTable`), and supervises satisfiable classes and conjunction pairs via signed non-emptiness (:class:`~mowl.nn.el.structural.ExistenceTargets`, :func:`~mowl.nn.el.structural.existence_loss`) and single-coordinate separation in ``gci1_bot``. At margin :math:`0` with ``structural_definitions=True``, zero sum of the structural box losses and the existence term implies that the materialised regions form a geometric model of the box normal forms with every satisfiable class and listed conjunction non-empty; the converse holds when every active non-empty source slot (in ``gci0``) or pair slot (in ``gci1``) is covered by a single active target slot (see the Non-guarantees below; role normal forms inherit Box²EL/PrismEL guarantees, and an all-empty target makes a role score vacuously :math:`0`).
     - :math:`K = \text{max\_components}` slots in :math:`\mathbb{R}^d` (:class:`~mowl.models.MultiBoxEL`, default :math:`k^2`) or single prisms in :math:`\mathbb{R}^2 \times \mathbb{R}^{d-2}` (:class:`~mowl.models.PrismEL`)


Limitations and caveats
-----------------------

1. **ABox nominal encoding in :class:`~mowl.geometric.completion.CanonicalModel`**

   * *Fragment for sound and complete instance checking:* :func:`~mowl.geometric.normal_forms.normalise`
     encodes each named individual :math:`a` as a fresh nominal concept name
     :math:`N_a = \text{nominal\_name}(a)` (prefix ``http://mowl.borg/nominal/``
     followed by the percent-encoded IRI), translating atomic ``ClassAssertion(C, a)``
     into ``NF1(N_a, C)`` and ``ObjectPropertyAssertion(r, a, b)`` into
     ``NF3(N_a, r, N_b)``. Instance checking
     (:meth:`~mowl.geometric.completion.CanonicalModel.instance_of`) and
     assertion contradiction
     (:meth:`~mowl.geometric.completion.CanonicalModel.assertion_contradicted`)
     are sound and complete for :math:`\mathcal{ELH}` (with :math:`\bot`)
     **without** nominals (``ObjectOneOf``), functionality
     (``FunctionalObjectProperty``), and range (``ObjectPropertyRange``) axioms.
   * *Separation of ``concept_names`` and ``names``:* :attr:`~mowl.geometric.completion.CanonicalModel.concept_names`
     excludes nominal names :math:`N_a`, whereas :attr:`~mowl.geometric.completion.CanonicalModel.names`
     keeps all satisfiable concept and nominal names so that constructive region
     builders assign every satisfiable individual a non-empty region.
   * *Inconsistent individuals and unsupported ABox axioms:* If completion
     derives :math:`N_a \sqsubseteq \bot` for any individual :math:`a`,
     constructing :class:`~mowl.geometric.completion.CanonicalModel` raises
     :class:`~mowl.geometric.errors.InconsistentOntology` naming :math:`a`.
     Unsupported ABox axioms (complex ``ClassAssertion``, ``SameIndividual``,
     ``DifferentIndividuals``, ``ObjectPropertyAssertion`` with an inverse
     property, assertions on anonymous individuals, and negative or data
     assertions) are never dropped silently and raise
     :class:`~mowl.geometric.errors.UnsupportedAxiomError` during
     :func:`~mowl.geometric.normal_forms.normalise`.

2. **Polyhedral dimension equals canonical-model size minus one (:math:`k = |\Delta^{\mathcal{I}}| - 1`)**
   In :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`, the ambient
   dimension is :math:`k = n - 1`, where :math:`n = |\text{cm.names}|` includes
   all satisfiable original concept names, nominal names, ``owl:Thing``, and any
   auxiliary concept names introduced by :class:`~mowl.ontology.normalize.ELNormalizer`.
   This is an upper bound: a dimension linear in the number of concept names
   is unavoidable in the worst case for convex models that realise the
   relevant conjunctions (Gutiérrez-Basulto & Schockaert, KR 2018, Prop. 4),
   but :math:`|\Delta^{\mathcal{I}}| - 1` itself, which counts auxiliary names
   and :math:`\top`, need not be minimal. LP checks over role polyhedra scale
   with :math:`n`.
   For large ontologies, prefer :class:`~mowl.geometric.intervals.IntervalUnionModel`.

3. **Polyhedral regions are lower-dimensional simplex faces unless ``thicken=True``**
   By default (``thicken=False``), each class region :math:`\eta(A)` in
   :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel` is the simplex face
   :math:`F(A^{\mathcal{I}}) = \text{conv}\{v_i : i \in A^{\mathcal{I}}\}`, which
   is non-empty for every satisfiable concept name but has zero Lebesgue volume
   (no interior in :math:`\mathbb{R}^{n-1}`) whenever :math:`|A^{\mathcal{I}}| < n`.
   This is required for exact satisfaction of role axioms (:class:`~mowl.geometric.normal_forms.NF3`
   and :class:`~mowl.geometric.normal_forms.NF4`). For *role-free*
   (propositional) ontologies, passing ``thicken=True`` replaces each face by a
   full-dimensional :class:`~mowl.geometric.regions.ThickFace` with non-empty
   interior (:math:`\varepsilon = \tfrac{1}{2n}`); passing ``thicken=True`` when
   role axioms are present raises :class:`~mowl.geometric.errors.UnsupportedAxiom`.

4. **Box builder is propositional-only, may be inconclusive, and produces zero-width boxes**

   * *Propositional fragment only:* :class:`~mowl.geometric.boxes.BoxModelBuilder`
     supports :class:`~mowl.geometric.normal_forms.NF1` and
     :class:`~mowl.geometric.normal_forms.NF2` (including :math:`\bot` forms). Any
     role axiom (:class:`~mowl.geometric.normal_forms.NF3`,
     :class:`~mowl.geometric.normal_forms.NF4`,
     :class:`~mowl.geometric.normal_forms.NF6`, or
     :class:`~mowl.geometric.normal_forms.NF7`) raises
     :class:`~mowl.geometric.errors.UnsupportedAxiom`.
   * *Which certificates reject:* in both ``faithful="weak"`` and
     ``faithful="strong"`` mode, the builder raises
     :class:`~mowl.geometric.errors.NotBoxRepresentable` only on a
     *non-degenerate* certificate. A strong certificate alone does not reject
     ``faithful="strong"``: it excludes conjunction-faithful box models, not
     box models strongly faithful on names. For example,
     :math:`\{X_1 \sqcap X_2 \sqsubseteq Y,\; Y \sqcap X_3 \sqsubseteq \bot\}` has
     the strong certificate :math:`(X_1, X_2, X_3)`, yet the 1-D boxes
     :math:`X_1 = [0, 1], X_2 = [2, 3], X_3 = [4, 5], Y = [6, 7]` are a
     non-degenerate model that is strongly faithful on names.
   * *Inconclusive MILP outcome:* A passing Helly check does *not* guarantee that
     an axis-parallel box model exists (no such sufficiency theorem is known);
     ``box_representable_* = True`` means only that the ontology is
     obstruction-free. Only a Helly certificate
     (:class:`~mowl.geometric.errors.NotBoxRepresentable`) is a proof of
     impossibility; when no certificate is found and the MILP fails for all
     :math:`d \le d_{\max}` or reaches ``time_limit`` without an incumbent that
     passes the post-check,
     :meth:`~mowl.geometric.boxes.BoxModelBuilder.build` raises
     :class:`~mowl.geometric.errors.BoxModelNotFound` (inconclusive). The
     builder is intended for small ontologies; the MILP is exponential in the
     worst case.
   * *Zero-width (degenerate) boxes:* Because the MILP objective minimises total
     box width :math:`\sum_{A,i} (U_{A,i} - L_{A,i})`, returned boxes frequently
     satisfy :math:`\text{lo} = \text{hi}` (single points, which are non-empty
     under closed-box semantics). Users exporting boxes via
     :func:`~mowl.geometric.export.boxes_as_tensors` to seed gradient-based
     models that require positive box volume should add a small positive margin
     to the exported offsets (or upper corners) or modify the objective.

5. **``HellyReport.complete`` and conservative ``box_representable_nondegenerate``**

   * ``HellyReport.complete`` is ``True`` if and only if the minimal jointly
     unsatisfiable set enumeration was not cut off by ``k_max`` or
     ``max_certificates`` **and** the ontology contains no role axioms
     (``has_roles == False``). Whenever role axioms are present, ``complete`` is
     ``False`` (Leemhuis & Kutz, NAI 2026, Prop. 38), though any returned
     certificate remains a sound proof of non-box-representability.
   * Non-degenerate certificates exclude every non-degenerate box model in any
     dimension. Strong certificates exclude only *conjunction-faithful* box
     models, in which every satisfiable conjunction of concept names is
     non-empty; this is stronger than strong faithfulness on names.
   * ``box_representable_nondegenerate`` / ``box_representable_strong`` equal to
     ``True`` means "no Helly obstruction of this kind was found"
     (obstruction-free), not that a box model exists. Build one with
     :class:`~mowl.geometric.boxes.BoxModelBuilder` to find out.
   * When ``complete=True``, the enumeration is exact for the **strong** reading
     (:attr:`~mowl.geometric.helly.HellyReport.box_representable_strong` /
     ``certificates_strong``). For the **non-degenerate** reading, whether a
     certificate is non-degenerate depends on how an unsatisfiable concept set is
     split into :math:`(C_1, C_2, C_3)`, and the union of names in a
     non-degenerate certificate need not itself be inclusion-minimal. Therefore
     :attr:`~mowl.geometric.helly.HellyReport.box_representable_nondegenerate` is
     conservative: it is ``False`` when ``certificates_nondegenerate`` is
     non-empty, ``True`` only when ``complete=True`` and ``certificates_strong``
     is empty, and ``None`` otherwise.
   * *Scaling:* :func:`~mowl.geometric.helly.helly_check` re-saturates the
     ontology per candidate set and is intended for small to medium
     ontologies; incremental joint satisfiability is future work.

6. **Lossy bounding-box export**
   :func:`~mowl.geometric.export.to_bounding_boxes` replaces each simplex face of
   a :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel` by its elementwise
   coordinate minimum and maximum. Because axis-parallel bounding boxes of
   simplex faces intersect strictly more often than the faces themselves, the
   resulting boxes **do not satisfy the ontology's disjointness or conjunction
   axioms** and must never be treated as a model.

7. **Projected box initialisation (:class:`~mowl.geometric.init.ExactBoxInit`) and ``certificate_triangles``**

   * *Satisfiable subclasses only; no disjointness under projected init:* With
     ``method="projected"`` and ``n_components=1``, :class:`~mowl.geometric.init.ExactBoxInit`
     places each canonical domain element at a Gaussian point in :math:`\mathbb{R}^d`
     and assigns each satisfiable concept name the padded bounding box of its
     extension. Because bounding boxes are monotone under set inclusion,
     :math:`\text{box}(C) \subseteq \text{box}(D)` holds (up to ``float32``
     rounding, ``1e-6``) for every entailed atomic subsumption :math:`C \sqsubseteq D`
     with a **satisfiable** subclass :math:`C` (giving zero ``gci0`` loss at
     margin :math:`0` immediately after :meth:`~mowl.geometric.init.ExactBoxInit.apply_to_module`).
     Axioms :math:`U \sqsubseteq A` with an **unsatisfiable** subclass :math:`U`
     are not covered (unsatisfiable concepts and ``BOTTOM`` are placed as point
     boxes at the origin, and no single point placement can satisfy subsumptions
     into mutually disjoint superclasses). Disjointness (:math:`A \sqcap B \sqsubseteq \bot`)
     is **not** guaranteed under ``method="projected"`` because projected
     bounding boxes overlap.
   * *Incompleteness of* :func:`~mowl.geometric.helly.certificate_triangles`:
     :func:`~mowl.geometric.helly.certificate_triangles` enumerates triangles in
     the graph of direct :class:`~mowl.geometric.normal_forms.NF2` definitions
     (:math:`A \sqcap B \sqsubseteq N` with :math:`\mathcal{O} \models N \equiv A \sqcap B`).
     Every returned certificate is a sound non-degenerate singleton
     :class:`~mowl.geometric.helly.HellyCertificate`, but the procedure is **not
     complete**: pairwise conjunctions defined through chains of equivalences or
     multi-step normalisations without a direct ``NF2(A, B, N)`` axiom, as well
     as higher-arity certificates where :math:`|C_i| > 1`, are not found.

8. **MultiBoxEL (:class:`~mowl.models.MultiBoxEL`, :class:`~mowl.nn.MultiBoxELModule`) and MultiELBE (:class:`~mowl.models.MultiELBE`, :class:`~mowl.nn.MultiELBEModule`)**

   * *Exact* :math:`k = 1` *identity:* When ``n_components=1``,
     ``component_aggregation="hard"``, ``component_reg=0.0``, and
     ``diversity_reg=0.0``, :class:`~mowl.nn.MultiBoxELModule` agrees with
     :class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule` and
     :class:`~mowl.nn.MultiELBEModule` agrees with
     :class:`~mowl.nn.el.elbe.module.ELBEModule` at equal ``margin`` to within
     ``1e-6`` on every GCI loss (positive and negative, including ``gci1``
     batches containing empty pairwise intersections). Note that
     :class:`~mowl.models.MultiELBE` defaults to ``margin=0.1`` (matching
     :class:`~mowl.nn.ELBEModule`), whereas the :class:`~mowl.models.ELBE`
     wrapper defaults to ``margin=0``, so comparing the model wrappers requires
     passing an equal ``margin`` explicitly.
   * *Gated pairwise intersections in* ``gci1`` *and overlap semantics in*
     ``MultiELBE.gci3``: For :math:`C \sqcap D \sqsubseteq E`, let
     :math:`(i^*, j^*) = \arg\min_{i,j} \text{empt}(C_i, D_j)` be the designated
     component pair (selected without gradient). Under ``"hard"`` aggregation,
     the detached gate :math:`g_{ij} = 1` if :math:`\text{lo}_{ij} \le \text{up}_{ij}`
     in every coordinate or :math:`(i, j) = (i^*, j^*)`, else :math:`0`, enforces
     inclusion into some component :math:`E_m` only on intersecting pairs (plus
     the designated pair, which also receives :math:`\text{empt}(C_{i^*}, D_{j^*})`).
     In :class:`~mowl.nn.MultiELBEModule`, ``gci3_loss`` preserves ELBE's
     coordinatewise overlap formula :math:`\|\text{relu}(|c - r - d| - o_c - o_d + \text{margin})\|_2`
     rather than full box inclusion so that :math:`k = 1` is identical to ELBE.
   * *Coordinatewise separation in* ``gci1_bot``: In both backbones, ``gci1_bot_loss``
     for :math:`C \sqcap D \sqsubseteq \bot` aggregates coordinatewise separation
     across all :math:`k^2` pairs. At margin :math:`0`, zero loss implies that
     all :math:`k^2` component pairs are disjoint (up to boundary contact), **but
     not conversely**: two boxes separated in one coordinate whose projections
     overlap in another coordinate are disjoint in :math:`\mathbb{R}^d` yet
     incur positive ``gci1_bot_loss``.
   * *Non-convexity:* Taking minima over target components and gating over
     pairwise intersections yields a piecewise-smooth non-convex objective;
     gradient-based training is not guaranteed to converge to a global optimum
     from arbitrary random initialisations.

9. **Soft component aggregation (:mod:`mowl.nn.el.softagg`) and canonical-element point supervision (:mod:`mowl.nn.el.pointloss`)**

   * *Finite-temperature collapse pressure under soft aggregation and ``tau_end=0.05`` floor:*
     At finite :math:`\tau > 0`, :func:`~mowl.nn.el.softagg.softmin` satisfies
     :math:`\min_j v_j \le \text{softmin}_\tau(v) \le \text{mean}_j v_j` and is
     strictly positive at a hard model whenever non-nearest target components have
     positive inclusion scores, pulling all target components together unless
     counteracted by ``diversity_reg > 0`` (:func:`~mowl.nn.el.softagg.diversity_penalty`).
     In ``float32``, softmax weights :math:`\exp(-\Delta/\tau)` underflow to
     exact ``0.0`` once the score gap :math:`\Delta` exceeds :math:`\approx 100\tau`;
     ``tau_end`` therefore defaults to ``0.05`` rather than ``0.01`` so
     non-nearest components retain non-zero gradient throughout annealing.
     Although ``module.tau`` is persisted as a registered buffer in
     ``state_dict()``, the geometric annealing schedule in ``on_epoch_start``
     restarts from ``tau_start`` at each :meth:`~mowl.models.MultiBoxEL.train` call.
   * *Strict positivity of BCE and region vs. point model-ness under point supervision:*
     Because :func:`~mowl.nn.el.pointloss.soft_box_membership` takes the minimum
     over :math:`d` coordinate sigmoids
     :math:`m(x) = \min_{\text{coord}} \sigma((|o| - |x - c|)/\beta)` (which is
     dimension-free, avoiding the :math:`\sigma(|o|/\beta)^d` decay of a
     coordinatewise product; individual targets in role edges use a virtual box
     of half-width ``ind_halfwidth`` defaulting to :math:`5\beta` around
     ``ind_embed``), at finite :math:`\beta > 0` and finite box half-widths
     every coordinate sigmoid is strictly in :math:`(0, 1)`, so
     :math:`0 < m(x) < 1` strictly (and is clamped to
     :math:`[\varepsilon, 1 - \varepsilon]` in
     :func:`~mowl.nn.el.pointloss.membership_bce`) and
     :func:`~mowl.nn.el.pointloss.membership_bce` never reaches exact zero.
     Under ``objective="point"``, all TBox GCI losses and ABox
     ``object_property_assertion`` are zeroed (while ABox ``class_assertion`` is
     still trained in the main loop via :math:`-\log m_C(a)`) because finite point
     sets do not forbid boxes from overlapping in empty space;
     ``objective="mixed"`` additionally keeps the backbone's :math:`\bot`-axiom
     losses (``gci0_bot``, ``gci1_bot``, ``gci3_bot``) alongside the point losses,
     and ``objective="hybrid"`` keeps all GCI and ABox losses alongside the point
     losses. Under ``"point"``, ``"mixed"``, and ``"hybrid"``, ABox class
     assertions are scored by :math:`-\log m_C(a)` and point satisfaction is
     measured via :meth:`~mowl.nn.MultiBoxELModule.point_modelness`.


Structural objective (learning Helly-breaking structure from random initialisation)
-----------------------------------------------------------------------------------

Standard inclusion and disjointness hinge losses have an incomplete zero set
because they are trivially satisfied by empty boxes, making shrinking a
universal descent direction with the degenerate all-empty interpretation as a
global minimum and nothing demanding that a satisfiable class or conjunction be
non-empty. Normalising a definition :math:`A \equiv X_1 \sqcap X_2` into three
separate hinges (:math:`A \sqsubseteq X_1`, :math:`A \sqsubseteq X_2`,
:math:`X_1 \sqcap X_2 \sqsubseteq A`) on an independently parametrised :math:`A`
leaves slack between :math:`A` and :math:`X_1 \cap X_2` where the :math:`T_3`
frustration (pushing :math:`A`, :math:`B`, :math:`C` apart vs. pulling :math:`A`
onto :math:`X_1 \cap X_2`) lives instead of forcing the parent boxes to split.
Because which component of :math:`X_1` realises which part of :math:`X_1` is
discrete, nearest-component aggregation gives gradient only to the component
already closest and a :math:`\operatorname{relu}(\text{overlap})` non-emptiness
term gives no gradient to a separated component pair, so descent keeps the
single-spanning-box assignment implied by random initialisation (where three
pairwise-intersecting spanning boxes always share a point by Helly's theorem).

The opt-in ``objective="structural"`` mode (implemented in
:mod:`mowl.nn.el.structural`, :mod:`mowl.nn.el.multiboxel`, and
:mod:`mowl.nn.el.prism`) addresses all three defects for
:class:`~mowl.models.MultiBoxEL` and :class:`~mowl.models.PrismEL`. The
guarantee concerns the zero set of the loss, not convergence of the optimiser;
as measured on the :math:`T_3` fixture in the slow tests (empirical measurements,
not a convergence guarantee; the synthetic sweep results live in the
experiments repository), :class:`~mowl.models.MultiBoxEL` (:math:`k = 4`,
:math:`K = 16`, :math:`d = 2`, re-seeding every 50 epochs) reached an exact,
strictly non-empty Helly-breaking model (``strict_loss == 0``, minimum target
class size :math:`\ge 0.0173 > 0`, minimum target pair size
:math:`\ge 0.0179 > 0`, and :math:`X_1 \cap X_2 \cap X_3 = \emptyset`) in 4 of
5 measured seeds and :class:`~mowl.models.PrismEL` (:math:`d = 3`) in 5 of 5
measured seeds (``strict_loss == 0``, minimum target class and pair
``prism_nonempty_measure`` :math:`\ge 0.0100 > 0`, and
:math:`X_1 \cap X_2 \cap X_3 = \emptyset`):

1. **Regions with signed half-widths and absent slots (:class:`~mowl.nn.el.structural.Regions`):**
   Every class :math:`C` is materialised as :math:`K = \text{max\_components}`
   component slots ``(cen, w, active)``, each a box :math:`(c, w)` with centre
   :math:`c \in \mathbb{R}^d`, **signed half-width** :math:`w \in \mathbb{R}^d`
   (negative in coordinate :math:`m` iff the slot is empty along coordinate
   :math:`m`), and boolean ``active`` flag:

   * *Primitive class:* slots ``0 .. k - 1`` hold the raw embedding rows
     (:math:`w = |o| \ge 0`, ``active = True``); slots ``k .. K - 1`` are
     inactive (``cen = 0``, ``w = -1``).
   * *Defined class* :math:`A := X \sqcap Y`: :func:`~mowl.nn.el.structural.pair_slots`
     forms all pairs :math:`(X_i, Y_j)` of parent slots with
     :math:`\text{lower} = \max(c_X - w_X, c_Y - w_Y)`,
     :math:`\text{upper} = \min(c_X + w_X, c_Y + w_Y)`,
     :math:`c = (\text{lower} + \text{upper})/2`, and
     :math:`w = (\text{upper} - \text{lower})/2` (negative in a coordinate
     :math:`\iff` the pair is disjoint there); a pair slot is active iff both
     parent slots are active, with active pairs packed into leading slots in
     lexicographic :math:`(i, j)` order and remaining slots padded as inactive
     (``cen = 0``, ``w = -1``).
   * *Derived differentiable quantities:*

     - :func:`~mowl.nn.el.structural.nonempty_measure`:
       :math:`\text{ne}(a) = \min_m 2 w_{a,m}` — minimum signed width; slot
       :math:`a` is non-empty iff :math:`\text{ne}(a) > 0`.
     - :func:`~mowl.nn.el.structural.signed_overlap`:
       :math:`\text{sov}_m(a, b) = \min(c_a + w_a, c_b + w_b)_m - \max(c_a - w_a, c_b - w_b)_m`
       — signed overlap per coordinate, with
       :math:`\text{ne}(a \cap b) = \min_m \text{sov}_m(a, b)`.
     - :func:`~mowl.nn.el.structural.region_size` and
       :func:`~mowl.nn.el.structural.pair_size`:
       :math:`\text{NE}(C) = \max_{i \text{ active}} \text{ne}(C_i)` and
       :math:`\text{NE}(C \sqcap D) = \max_{(i,j) \text{ active}} \text{ne}(C_i \cap D_j)`.
       The maximum selects the nearest active pair even when all pairs are
       separated (negative values), routing gradient to pull separated
       components together.
     - :func:`~mowl.nn.el.structural.inclusion_tight`:
       :math:`\text{incl}_m(a, b) = \|\operatorname{relu}(|c_a - c_b| + w_a - w_b + m)\|_2`
       — tight inclusion with margin :math:`m \ge 0` (positive :math:`m`
       tightens, ELBE convention), equal to :math:`0` at :math:`m = 0` iff
       :math:`a \subseteq b` coordinatewise (for non-empty :math:`a`).
   * *Absent-slot export convention:* A slot that is inactive
     (``active == False``) or has :math:`w_m < 0` for some coordinate :math:`m`
     is **absent**. :meth:`~mowl.nn.MultiBoxELModule.class_regions` and
     :meth:`~mowl.nn.MultiBoxELModule.all_class_regions` return ``w = -1`` in
     every coordinate and ``cen = 0`` for inactive slots; consumers exporting
     ``(centers, offsets)`` arrays mark any absent slot
     (:math:`\neg\text{active} \vee \min_m w_m < 0`) with ``offsets = -1`` and
     ``centers = 0``, so standard corner arithmetic
     (:math:`\text{lo} = c - o > \text{hi} = c + o`) treats absent slots as
     empty automatically.

2. **Definition table (:class:`~mowl.nn.el.structural.DefinitionTable`):**
   Built once per :meth:`~mowl.models.MultiBoxEL.train` /
   :meth:`~mowl.models.PrismEL.train` call via
   :meth:`~mowl.nn.el.structural.DefinitionTable.from_normal_forms` from the
   training normal forms (``gci0`` rows :math:`(C, D)`, ``gci1`` rows
   :math:`(C, D, E)`, and optional ``gci2`` rows :math:`(C, r, D)` and
   ``gci3`` rows :math:`(r, C, D)`):

   * *Candidate (including role auxiliaries):* :math:`A` is a candidate defined
     class with definers :math:`(X, Y)` iff there is a ``gci1`` row
     :math:`(X, Y, A)` with :math:`X \neq Y`, :math:`A \notin \{X, Y\}`, none
     of :math:`A, X, Y` equal to ``owl:Thing`` or ``owl:Nothing``, and each
     required inclusion :math:`(A, D)` for :math:`D \in \{X, Y\}` is satisfied
     either directly by a ``gci0`` row :math:`(A, D)` or through a role
     auxiliary by a ``gci2`` row :math:`(A, r, C)` together with a ``gci3`` row
     :math:`(r, C, D)` sharing the same :math:`r` and :math:`C` (i.e.
     :math:`A \sqsubseteq \exists r.C \sqsubseteq D` is entailed, which is how
     :class:`~mowl.ontology.normalize.ELNormalizer` normalises definitions of
     the form :math:`A \equiv X \sqcap \exists r.C` into :math:`A \sqsubseteq X`,
     :math:`A \sqsubseteq \exists r.C`, :math:`\exists r.C \sqsubseteq \text{Aux}`,
     and :math:`X \sqcap \text{Aux} \sqsubseteq A`; definitions whose
     normalisation introduces nested auxiliaries on both sides—e.g.
     :math:`A \equiv \text{Pizza} \sqcap \exists\text{hasTopping}.(\text{PizzaTopping} \sqcap \exists\text{hasSpiciness}.\text{Hot})`,
     which yields :math:`A \sqsubseteq \exists r.\text{Aux}_1` and
     :math:`\exists r.\text{Aux}_2 \sqsubseteq \text{Aux}_3` with
     :math:`\text{Aux}_1 \neq \text{Aux}_2`—are not detected, only a single
     role-auxiliary step :math:`A \sqsubseteq \exists r.C \sqsubseteq \text{Aux}`
     is). If several :math:`(X, Y)`
     qualify for the same :math:`A`, the lexicographically smallest
     :math:`(\min(X, Y), \max(X, Y))` is selected.
   * *Bottom-up acceptance and cap:* Candidates are processed in topological
     order of the candidate dependency graph (a candidate is ready once its
     definers are decided), with ties broken by increasing class index (Kahn's
     algorithm with a min-heap in :math:`O((|G_0| + |G_1|) \log n)` time). A
     ready candidate :math:`A \to (X, Y)` is accepted iff
     :math:`\text{comps}(X) \cdot \text{comps}(Y) \le \text{max\_components}`
     (and :math:`\text{level}(A) \le \text{max\_level}` when given), where
     :math:`\text{comps}(P) = k` for primitive classes and
     :math:`\text{comps}(X) \cdot \text{comps}(Y)` for accepted defined classes,
     and :math:`\text{level}(A) = 1 + \max(\text{level}(X), \text{level}(Y))`
     (:math:`\text{level} = 0` for primitive classes). Because definers are
     final at acceptance time, nothing accepted earlier can be invalidated.
     When only cyclic (or cycle-blocked) candidates remain, the undecided
     candidate with the lowest class index is rejected (stays primitive) and
     processing resumes (deterministic, though possibly lossy because the
     lowest-index undecided candidate may sit downstream of a cycle rather than
     on it). A
     level-:math:`\ell` class has at most :math:`k^{2^\ell}` slots (when both
     definers are at level :math:`\ell - 1`) and :math:`k^{\ell+1}` slots for a
     chained definition :math:`A \equiv \text{Aux} \sqcap X` (the shape emitted
     by :class:`~mowl.ontology.normalize.ELNormalizer` for :math:`n`-ary
     conjunctions); ``max_components`` (default :math:`k^2`, one level) bounds
     this blow-up.
   * *Masked rows:* For every accepted :math:`A \to (X, Y)`, the rows
     :math:`(A, X)` and :math:`(A, Y)` in ``gci0``
     (:meth:`~mowl.nn.el.structural.DefinitionTable.masked_gci0`) and
     :math:`(X, Y, A)` and :math:`(Y, X, A)` in ``gci1``
     (:meth:`~mowl.nn.el.structural.DefinitionTable.masked_gci1`) hold by
     construction and evaluate to exact ``0.0`` loss in training and evaluation;
     when :math:`(A, \text{Aux})` is only entailed through the role auxiliary,
     no ``gci0`` row exists for it and therefore only :math:`(A, X)` and the
     ``gci1`` rows are masked, while the ``gci2`` and ``gci3`` rows are **not**
     masked and keep training :math:`\text{Aux}`'s region.
     Setting ``structural_definitions=False`` disables the table (every class
     stays primitive and nothing is masked).

3. **Structural losses (:class:`~mowl.models.MultiBoxEL`) and existence / size terms:**
   All box normal-form losses are per axiom row, squared hinges with tightening
   margin :math:`m` (``module.margin``, default ``0.02`` on
   :class:`~mowl.models.MultiBoxEL` and ``0.1`` on
   :class:`~mowl.models.PrismEL`), where :math:`[\cdots]^{\text{det}}`
   denotes a detached gate and means/minima are over active slots:

   .. list-table:: Structural normal-form losses for :class:`~mowl.nn.MultiBoxELModule`
      :header-rows: 1
      :widths: 28 72

      * - Normal form
        - Structural loss (per row)
      * - :math:`C \sqsubseteq D` (``gci0``, :func:`~mowl.nn.el.structural.gci0_structural`)
        - :math:`\operatorname{mean}_{i:\,\text{active},\,\text{ne}(C_i)>0}^{\text{det}} \min_{j \text{ active}} \text{incl}_m(C_i, D_j)^2`; ``0`` if no active non-empty :math:`C_i` or if the row is masked
      * - :math:`C \sqsubseteq \bot` (``gci0_bot``, :func:`~mowl.nn.el.structural.gci0_bot_structural`)
        - :math:`\operatorname{mean}_{i \text{ active}} \operatorname{relu}(\text{ne}(C_i))^2` (no margin: primitive widths satisfy :math:`w = |o| \ge 0`, so a positive margin would impose a floor on the loss)
      * - :math:`C \sqcap D \sqsubseteq E` (``gci1``, :func:`~mowl.nn.el.structural.gci1_structural`)
        - :math:`\operatorname{mean}_{(i,j):\,\text{active},\,\text{ne}(C_i \cap D_j)>0}^{\text{det}} \min_{l \text{ active}} \text{incl}_m(I_{ij}, E_l)^2` with :math:`I_{ij}` the pair slot; ``0`` if no pair is non-empty (vacuous; no emptiness penalty) or the row is masked
      * - :math:`C \sqcap D \sqsubseteq \bot` (``gci1_bot``, :func:`~mowl.nn.el.structural.gci1_bot_structural`)
        - :math:`\operatorname{agg}_{(i,j) \text{ active}} \operatorname{relu}(\text{ne}(C_i \cap D_j) + m)^2` with ``agg`` = ``bot_agg`` :math:`\in \{\text{"mean"}, \text{"sum"}\}` (default ``"mean"``; ``"sum"`` sums over active pairs instead of dividing by the active-pair count, with module slots gated to active non-empty components :math:`\text{ne} > 0`: identical zero set at :math:`m = 0`, deviating from gating on ``active`` alone only at :math:`m > 0`) — separation by :math:`m` in **at least one** coordinate
      * - :math:`C \sqsubseteq \exists r.D`, :math:`\exists r.C \sqsubseteq D`, :math:`\exists r.C \sqsubseteq \bot`, and ABox assertions (``gci2``, ``gci3``, ``gci3_bot``, ``class_assertion``, ``object_property_assertion``)
        - Existing Box²EL multibox scores (hard aggregation, with ``gci3`` and ``gci3_bot`` squared) evaluated on the materialised slots :math:`(c, |w|)` with inactive or empty slots (:math:`\neg\text{active} \vee \text{ne}(w) \le 0`) masked out (source slots excluded from means, target slots excluded from minima); because Box²EL ``gamma`` loosens inclusions (:math:`\operatorname{relu}(\text{dist} + 2o - \gamma)`), positive role and assertion scores pass ``gamma = -m`` (tightening inclusion by :math:`m`), whereas negative scores (:func:`~mowl.nn.el.multiboxel.losses.gci2_score_neg` and negative ``object_property_assertion``) keep Box²EL's convention (``gamma = +margin``, which relaxes them by the margin), so the negative-sampling term is unchanged between the axiom and structural objectives; class bumps remain per-class embedding rows

   Additional per-epoch terms returned by
   :meth:`~mowl.models.MultiBoxEL.additional_training_loss` via
   :meth:`~mowl.nn.MultiBoxELModule.structural_extra_loss`:

   * *Existence* (weight ``existence_weight``, default ``1.0``,
     :func:`~mowl.nn.el.structural.existence_loss`):

     .. math::

        \operatorname{mean}_{C \in \text{Sat}} \operatorname{relu}(m - \text{NE}(C))^2
        + \operatorname{mean}_{(C, D) \in \text{SatPairs}} \operatorname{relu}(m - \text{NE}(C \sqcap D))^2,

     where :class:`~mowl.nn.el.structural.ExistenceTargets` collects ``Sat``
     (satisfiable classes excluding :math:`\top, \bot`) and ``SatPairs``
     (distinct unordered pairs :math:`(C, D)` occurring in ``gci1`` rows with
     :math:`C \sqcap D` satisfiable). When a ``canonical_model``
     (:class:`~mowl.geometric.completion.CanonicalModel`) is passed,
     satisfiability for IRIs in ``canonical_model.concept_names`` comes from
     :meth:`~mowl.geometric.completion.CanonicalModel.is_satisfiable` and
     :meth:`~mowl.geometric.completion.CanonicalModel.joint_satisfiable`, while
     IRIs unknown to ``canonical_model.concept_names`` fall back to the asserted
     ``gci0_bot``/``gci1_bot`` rules for both single classes and pairs (note
     that auxiliary classes minted by :class:`~mowl.ontology.normalize.ELNormalizer`,
     ``http://mowl.borg/aux/...``, are unknown to a
     :class:`~mowl.geometric.completion.CanonicalModel` built with
     ``mowl.geometric``'s own normaliser, so their entailed unsatisfiability
     cannot be detected via ``canonical_model`` and uses the asserted rules
     only); otherwise a class is assumed satisfiable unless a ``gci0_bot`` row
     :math:`(C, \bot)` exists and a ``gci1`` pair unless either class appears in
     ``gci0_bot`` or :math:`(C, D)` appears in ``gci1_bot``.
   * *Size* (weight ``size_reg``, default ``1e-3``): mean of :math:`2|o|` over
     raw primitive slots and coordinates (excluding :math:`\top` and accepted
     defined classes), keeping spare components small so they can be relocated
     and discouraging single spanning boxes.
   * *Witness points* (weight ``witness_weight``, default ``0.0``,
     :func:`~mowl.nn.el.structural.witness_loss`): when ``witness_weight > 0``,
     allocates ``module.witness_embed`` with one learnable witness point
     :math:`w_C \in \mathbb{R}^d` per satisfiable class
     :math:`C \in \text{targets.witness.classes}` (:class:`~mowl.nn.el.structural.WitnessTargets`)
     and adds

     .. math::

        \operatorname{mean}_{(C, D) \in \text{pos}} \operatorname{relu}(m - \text{inside}(w_C, D))^2
        + \operatorname{mean}_{(C, D) \in \text{neg}} \operatorname{relu}(\text{inside}(w_C, D) + m)^2,

     where :func:`~mowl.nn.el.structural.inside_measure` evaluates
     :math:`\text{inside}(p, D) = \max_{s \text{ active}} \min_m (w_{D,s,m} - |p_m - c_{D,s,m}|)`
     on the materialised :meth:`~mowl.nn.MultiBoxELModule.class_regions` of
     :math:`D` (returning ``-1e6`` if no slot is active), ``pos`` collects
     :math:`(w_C, D)` with :math:`C \sqsubseteq D` (via
     :meth:`~mowl.geometric.completion.CanonicalModel.entails` when
     ``canonical_model`` is given, or :math:`\{C\} \cup \{D : (C, D) \in \text{gci0}\}`
     otherwise), and ``neg`` collects satisfiable :math:`D` disjoint from
     :math:`C` (via ``not canonical_model.joint_satisfiable([C, D])`` when
     ``canonical_model`` is given, or asserted ``gci1_bot`` partners otherwise),
     excluding :math:`\top` and :math:`\bot` throughout. Witness coordinates can
     be inspected via :meth:`~mowl.nn.MultiBoxELModule.witness_points`.

   * **Guarantee:** With ``structural_definitions=True``, at margin :math:`m = 0`,
     zero sum of the structural box losses (``gci0``, ``gci0_bot``, ``gci1``,
     ``gci1_bot``) and the existence term implies that the materialised regions
     form a geometric model of the box normal forms in which every satisfiable
     class and listed conjunction is non-empty (up to boundary contact
     :math:`w_m = 0`); the converse holds when every active non-empty source
     slot (in ``gci0``) or pair slot (in ``gci1``) is covered by a single active
     target slot (see the Non-guarantees bullet below), while role normal forms
     inherit the Box²EL guarantees.
   * **Non-guarantees:** In ``gci2``, ``gci3``, and ``class_assertion``, masking
     out empty target slots means that an **all-empty target** (a target class
     :math:`D` with no active non-empty slot) makes the target minimum fall back
     to ``0.0`` and renders the role or assertion score (or the target half of
     ``gci2``) vacuously ``0`` unless the existence term forces :math:`D` to be
     non-empty; similarly, in the negative score
     :func:`~mowl.nn.el.multiboxel.losses.gci2_score_neg`, an all-masked (empty)
     source :math:`C` makes the source half return ``0.0`` rather than
     :math:`\delta^2` (and ``0.0`` overall when both source and target are
     all-masked). Because :math:`\top` (``owl:Thing``) is excluded from
     :class:`~mowl.nn.el.structural.ExistenceTargets`, axioms with :math:`\top`
     on the left-hand side are vacuous if :math:`\top` collapses to an empty
     region. Furthermore, :func:`~mowl.nn.el.structural.region_size` and
     :func:`~mowl.nn.el.structural.pair_size` route gradient only through the
     argmax active slot or pair, and each non-empty source slot :math:`C_i` in
     ``gci0`` or pair slot :math:`C_i \cap D_j` in ``gci1`` must be covered by a
     single active target slot rather than a union of several target slots.

4. **Re-seeding (:meth:`~mowl.nn.MultiBoxELModule.reseed`):**
   Controlled by ``reseed_every`` (``reseed_every=0`` is the default, which
   disables re-seeding), ``reseed_mode`` (``{"dead", "idle"}``, default
   ``"dead"``), ``reseed_size`` (``float = 0.15``), and ``reseed_batch``
   (``int = 1``); re-seeding is a heuristic discrete move rather than part of
   the continuous objective. When ``reseed_every > 0`` under
   ``objective="structural"``, at the start of every ``reseed_every``-th epoch
   (``epoch > 0``), :meth:`~mowl.models.MultiBoxEL.on_epoch_start` invokes
   :meth:`~mowl.nn.MultiBoxELModule.reseed` under ``torch.no_grad()`` on the raw
   embedding rows of **primitive** classes:

   * ``reseed_mode="dead"`` (default):

     - A primitive component :math:`i` of :math:`C` is classified as **dead** if
       it is nested inside another active valid component :math:`j` of :math:`C`
       (:math:`\text{lo}_j \le \text{lo}_i` and :math:`\text{hi}_i \le \text{hi}_j`
       in every coordinate, exact ties resolved by lower index) or has
       :math:`\min_m |o_{i,m}| < 10^{-3}`.
     - Violated pair targets :math:`(C, D)` (:math:`\text{NE}(C \sqcap D) < m`)
       are processed in a seeded random order (seed ``points_seed + epoch``),
       using each dead component at most once per epoch: take a dead component of
       :math:`C` if available, else of :math:`D`; let :math:`P` be the partner
       class and :math:`P_j` its active slot with largest :math:`\text{ne}(P_j) > 0`,
       and set :math:`c_i := c_{P_j}`, :math:`o_i := 0.5 \cdot |w_{P_j}|`.
     - Violated primitive class targets (:math:`\text{NE}(C) < m`) have the
       offset of their largest-measure component clamped to :math:`\max(|o|, m)`.
     - The number of moves is stored in ``module.last_reseed_count``; optimiser
       moments of moved rows are not reset.

   * ``reseed_mode="idle"`` (requires ``witness_weight > 0``, raising
     :class:`ValueError` otherwise, and uses ``reseed_size: float = 0.15`` and
     ``reseed_batch: int = 1``):

     1. **Dead defined classes**: accepted defined :math:`A` with
        :math:`\text{NE}(A) = \max_{s \text{ active}} \text{ne}(A_s) < m`.
     2. **Usage counts** :math:`u[X, c]` over primitive :math:`X` and component
        :math:`c`: number of accepted defined classes :math:`A'` with
        :math:`\text{NE}(A') \ge m` whose best slot (:math:`\arg\max \text{ne}`)
        uses component :math:`c` of :math:`X`, where the slot-to-component map
        (:func:`~mowl.nn.el.structural.slot_components`) follows the recursion
        of :meth:`~mowl.nn.MultiBoxELModule.class_regions` /
        :func:`~mowl.nn.el.structural.pair_slots` (a slot of
        :math:`A \to (X, Y)` is a pair of a slot of :math:`X` and a slot of
        :math:`Y`, recursing to primitives).
     3. Process dead defined classes in a seeded random order
        (``rng.permutation``), at most ``reseed_batch`` per call. For :math:`A`
        with witness :math:`p`: for each primitive leaf :math:`X` of :math:`A`'s
        definition tree (deduplicated): if some active component :math:`c` of
        :math:`X` has :math:`\text{inside}(p, X_c) \ge m`, skip :math:`X`; else
        :math:`c^* = \arg\min_c u[X, c]` (lowest index on ties),
        :math:`\text{center}[X, c^*] := p + 0.01 \cdot \mathcal{N}(0, I)`
        (seeded through ``rng``),
        :math:`\text{offset}[X, c^*] := \text{reseed\_size}` in every
        coordinate, :math:`u[X, c^*] \mathrel{+}= 1`. Count one move per
        re-seeded component.
     4. Keep step 3 of the ``"dead"`` mode (clamp the largest component of a
        primitive class target with :math:`\text{NE} < m` to
        :math:`\max(|o|, m)`).
     5. ``module.last_reseed_count`` as before; optimiser moments are not reset.

5. **Prisms (:class:`~mowl.models.PrismEL`, :class:`~mowl.nn.PrismModule`):**
   Under ``PrismEL(..., margin=0.1, objective="structural", max_depth=8, structural_definitions=True, existence_weight=1.0, size_reg=1e-3, canonical_model=None)``,
   each class is materialised as a single prism
   :class:`~mowl.nn.el.prism.losses.PrismRegions` ``(h, cen, w)`` with support
   vector :math:`h \in \mathbb{R}^3`, box centre :math:`\text{cen} \in \mathbb{R}^{d-2}`,
   and signed box half-width :math:`w \in \mathbb{R}^{d-2}`. Non-emptiness is
   measured by :func:`~mowl.nn.el.prism.losses.prism_nonempty_measure`:
   :math:`\text{ne}(P) = \min(\sum_{k=1}^3 h_k,\; \min_m 2 w_m)` (strictly
   positive iff non-empty; :math:`\sum_{k=1}^3 h_k = 3|\rho| \ge 0` for
   primitive prisms). Definitions are evaluated recursively with
   :func:`~mowl.nn.el.prism.losses.prism_regions_intersection`
   (:math:`h = \min(h_X, h_Y)` and signed box half-widths from
   :func:`~mowl.nn.el.prism.losses.prism_intersection`); because intersecting
   two prisms yields a single prism, there is no slot blow-up (:math:`k = 1`)
   and acceptance only requires acyclicity and :math:`\text{level} \le \text{max\_depth}`.

   .. list-table:: Structural normal-form losses for :class:`~mowl.nn.PrismModule`
      :header-rows: 1
      :widths: 28 72

      * - Normal form
        - Structural prism loss (per row)
      * - :math:`C \sqsubseteq D` (``gci0``, :func:`~mowl.nn.el.prism.losses.gci0_structural`)
        - :math:`[\text{ne}(C) > 0]^{\text{det}} \bigl(\|\operatorname{relu}(h_C - h_D + m)\|_2^2 + \|\operatorname{relu}(|\text{cen}_C - \text{cen}_D| + w_C - w_D + m)\|_2^2\bigr)` on signed widths :math:`w_C, w_D` (equivalent to :math:`\text{prism\_inclusion\_score}(C, D, m)^2` for non-negative widths; masked rows ``0``)
      * - :math:`C \sqsubseteq \bot` (``gci0_bot``, :func:`~mowl.nn.el.prism.losses.gci0_bot_structural`)
        - :math:`\operatorname{relu}(\text{ne}(C))^2`
      * - :math:`C \sqcap D \sqsubseteq E` (``gci1``, :func:`~mowl.nn.el.prism.losses.gci1_structural`)
        - :math:`[\text{ne}(I) > 0]^{\text{det}} \bigl(\|\operatorname{relu}(h_I - h_E + m)\|_2^2 + \|\operatorname{relu}(|\text{cen}_I - \text{cen}_E| + w_I - w_E + m)\|_2^2\bigr)` with :math:`I = C \cap D` on signed widths :math:`w_I, w_E` (equivalent to :math:`\text{prism\_inclusion\_score}(C \cap D, E, m)^2` for non-negative widths), masked rows ``0``, and **no** emptiness penalties on :math:`C \cap D` (the existence term owns them)
      * - :math:`C \sqcap D \sqsubseteq \bot` (``gci1_bot``, :func:`~mowl.nn.el.prism.losses.gci1_bot_structural`)
        - :math:`\operatorname{relu}(\text{ne}(C \cap D) + m)^2`
      * - ``gci2``, ``gci3``, ``gci3_bot``, and assertions
        - Existing prism scores (:func:`~mowl.nn.el.prism.losses.gci2_score_prisms`, :func:`~mowl.nn.el.prism.losses.gci2_score_neg_prisms`, :func:`~mowl.nn.el.prism.losses.gci3_score_prisms`, :func:`~mowl.nn.el.prism.losses.gci3_bot_score_prisms`, :func:`~mowl.nn.el.prism.losses.class_assertion_score_prisms`, :func:`~mowl.nn.el.prism.losses.object_property_assertion_loss`) on materialised prisms with ``|w|`` (unsquared norms/hinges)

   The existence term uses :math:`\text{NE}(C) = \text{ne}(C)` and
   :math:`\text{NE}(C \sqcap D) = \text{ne}(C \cap D)`, and the size term
   averages :math:`|\rho|` and :math:`2|o|` over primitive classes (excluding
   :math:`\top`). There is no re-seeding (single component). Under
   ``objective="structural"``, :meth:`PrismEL.compute_loss <mowl.models.PrismEL.compute_loss>`
   returns ``pos_scores.mean()`` directly (without an outer squaring): only the
   box structural GCI losses (``gci0``, ``gci0_bot``, ``gci1``, ``gci1_bot``)
   are squared hinges, whereas the prism role and assertion scores (``gci2``,
   ``gci3``, ``gci3_bot``, ``class_assertion``, ``object_property_assertion``)
   are unsquared norms/hinges that enter linearly (while still adding
   ``MSE(neg_scores.mean(), 1)`` when ``neg_scores`` is present).

   * **Prism non-guarantees:** Because the prism role and assertion scores take
     :math:`|w|` on materialised defined classes, a defined class that is empty
     only in its box part (:math:`w_m < 0` in some coordinate while
     :math:`\sum_k h_k > 0`) becomes a phantom non-empty box of half-width
     :math:`|w_m| > 0` in ``gci2``, ``gci3``, and ``class_assertion``. In
     addition, :func:`~mowl.nn.el.prism.losses.gci3_bot_score_prisms`
     (:math:`\operatorname{relu}(\sum_k h_{C,k} + m)`) shrinks :math:`C` itself,
     which conflicts with the existence term whenever :math:`C` is satisfiable.

6. **Public interface:**

   * ``MultiBoxEL(dataset, ..., margin=0.02, objective="structural", max_components=None, structural_definitions=True, existence_weight=1.0, size_reg=1e-3, reseed_every=0, canonical_model=None, bot_agg="mean", witness_weight=0.0, reseed_mode="dead", reseed_size=0.15, reseed_batch=1)``
     (where ``max_components=None`` defaults to :math:`k^2`, ``bot_agg`` is
     ``"mean"`` or ``"sum"``, ``reseed_mode`` is ``"dead"`` or ``"idle"``, and
     ``component_aggregation`` must be ``"hard"``).
   * ``PrismEL(dataset, ..., margin=0.1, objective="structural", max_depth=8, structural_definitions=True, existence_weight=1.0, size_reg=1e-3, canonical_model=None)``.
   * Default ``objective="axiom"`` leaves every existing code path unchanged.
     Negative sampling stays enabled under ``objective="structural"`` just as in
     the ``objective="axiom"`` path.
   * Module region and table inspection:
     :meth:`MultiBoxELModule.class_regions(idx) <mowl.nn.MultiBoxELModule.class_regions>`
     :math:`\to` :class:`~mowl.nn.el.structural.Regions` ``(cen (B, K, d), w (B, K, d), active (B, K))``
     and :meth:`MultiBoxELModule.all_class_regions() <mowl.nn.MultiBoxELModule.all_class_regions>`
     (detached on CPU);
     :meth:`PrismModule.class_regions(idx) <mowl.nn.PrismModule.class_regions>`
     :math:`\to` :class:`~mowl.nn.el.prism.losses.PrismRegions` ``(h (B, 3), cen (B, d - 2), w (B, d - 2))``
     and :meth:`PrismModule.all_class_regions() <mowl.nn.PrismModule.all_class_regions>`
     (detached on CPU); ``module.definition_table``
     (:class:`~mowl.nn.el.structural.DefinitionTable`),
     ``module.existence_targets``
     (:class:`~mowl.nn.el.structural.ExistenceTargets`), ``module.top_idx``, and
     ``module.margin`` (tightening margin under both models when
     ``objective="structural"``, passed as ``gamma = -margin`` to positive
     Box²EL role/assertion scores and ``gamma = +margin`` to negative Box²EL
     role/assertion scores: default ``0.02`` on :class:`~mowl.models.MultiBoxEL`
     and ``0.1`` on :class:`~mowl.models.PrismEL`). Both models build
     ``definition_table`` and ``existence_targets`` via
     :func:`~mowl.nn.el.structural.build_structural_tables` at the start of
     ``train()``, cache them on the model, and preserve them across
     ``init_module()``, ``load_best_model()``, and ``from_pretrained()``.

   **Recommended settings for learning from random initialisation:** For
   learning Helly-breaking multi-component box models from random
   initialisation (including deep-chain theories where all live definitions
   otherwise share the same primitive components), use
   ``objective="structural", bot_agg="sum", witness_weight=1.0, reseed_every=50..100, reseed_mode="idle", size_reg=0.0``.


References
----------

* V. Gutiérrez-Basulto and S. Schockaert, *From Knowledge Graph Embedding to
  Ontology Embedding? An Analysis of the Compatibility between Vector Space
  Representations and Rules*, in **Proceedings of KR 2018** (arXiv:1805.10461),
  Propositions 3 and 4.
* V. Lacerda, A. Ozaki, and R. Guimarães, *Strong Faithfulness for ELH
  Ontology Embeddings*, **Transactions on Graph Data and Knowledge (TGDK)**
  2(3):2, 2024, Definitions 2, 8, 12 and Theorems 13, 16, 18, 25.
* M. Leemhuis and O. Kutz, *Understanding the Expressive Capabilities of
  Knowledge Base Embeddings under Box Semantics*, in **Proceedings of NeSy
  2025** (PMLR 284:303–321), Definitions 1, 3, 6 and Propositions 4, 5, 8.
* M. Leemhuis and O. Kutz, *Trustworthy Knowledge Base Embeddings: A
  Foundational Study of Box Semantics*, **Neurosymbolic Artificial
  Intelligence** (NAI), 2026, Proposition 38 (incompleteness of ABox-closure
  Helly tests in the presence of roles).
* G. Federico, M. Leemhuis, and O. Kutz, *Introducing Multiboxes: Increasing
  the Trustworthiness of Ontology Embeddings*, in **Proceedings of JOWO 2026**
  (OSETAI workshop), Theorem 3 (1-D case).
* G. Federico, M. Leemhuis, and O. Kutz, *Introducing Prism Embeddings*, in
  **Proceedings of DL 2026**, §3–4.
* C. Bourgaux, R. Guimarães, R. Koudijs, V. Lacerda, and A. Ozaki, *Knowledge
  Base Embeddings: Semantics and Theoretical Properties*, in **Proceedings of
  KR 2024**, Definitions 5–6 and Remark 2.
* Y. Kazakov, M. Krötzsch, and F. Simančík, *The Incredible ELK: From Polynomial
  Procedures to Efficient Reasoning with EL Ontologies*, **Journal of Automated
  Reasoning** 53(1):1–61, 2014.
* F. Baader, S. Brandt, and C. Lutz, *Pushing the EL Envelope*, in
  **Proceedings of IJCAI 2005**.
* J. Eckhoff, *Intersection properties of boxes. Part I: An upper-bound
  theorem*, **Israel Journal of Mathematics** 62(3):283–301, 1988.


Development setup
-----------------

To work on :mod:`mowl.geometric` from a source checkout (Python 3.8–3.11; Java
and Gradle for the gateway jars):

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip wheel
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -e .
   pip install "class-resolver<0.5.0"   # needed by pykeen 1.10.1
   pip install pytest
   ./build_jars.sh                      # builds mowl/lib/*.jar
   python -m pytest -m "not slow" -p no:faulthandler tests/geometric -q

Slow tests (e.g. the randomised differential test against ELK) run without
``-m "not slow"``.


.. toctree::
   :maxdepth: 2

   examples
   api
