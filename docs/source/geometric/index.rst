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

The :mod:`mowl.geometric` package implements these constructive models, an exact
geometric model checker (:class:`~mowl.geometric.checker.ModelChecker`), a
Helly-satisfiability diagnostic with non-box-representability certificates
(:func:`~mowl.geometric.helly.helly_check` and the scalable triangle finder
:func:`~mowl.geometric.helly.certificate_triangles`), a mixed-integer linear
programming builder for axis-parallel box models
(:class:`~mowl.geometric.boxes.BoxModelBuilder`), and exact-model box
initialisation and anchoring for learnable box embeddings
(:class:`~mowl.geometric.init.ExactBoxInit`, integrated with
:class:`~mowl.models.BoxSquaredEL`, :class:`~mowl.models.ELBE`, and the
multi-component :class:`~mowl.models.MultiBoxEL` model).


Guarantees by model and procedure
---------------------------------

Every constructed model (:class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`,
:class:`~mowl.geometric.intervals.IntervalUnionModel`, and
:class:`~mowl.geometric.boxes.BoxModel`) is verified by
:class:`~mowl.geometric.checker.ModelChecker` before it is returned; if a
constructed model ever fails verification, construction raises
:class:`~mowl.geometric.errors.InternalConsistencyError`.

.. list-table:: Constructive models and diagnostics in ``mowl.geometric``
   :header-rows: 1
   :widths: 22 20 38 20

   * - Class / function
     - Supported fragment
     - Mathematical guarantee
     - Dimension
   * - :class:`~mowl.geometric.completion.CanonicalModel`
     - Normalised :math:`\mathcal{EL}^{++}` (:class:`~mowl.geometric.normal_forms.NF1`–:class:`~mowl.geometric.normal_forms.NF4`, :class:`~mowl.geometric.normal_forms.NF6`, :class:`~mowl.geometric.normal_forms.NF7`, and :math:`\bot` forms)
     - Sound and complete on concept-name subsumptions and unsatisfiability wrt the normalised axiom set (`Kazakov et al., JAR 2014`). Raises :class:`~mowl.geometric.errors.InconsistentOntology` iff :math:`\mathcal{O} \models \top \sqsubseteq \bot`.
     - :math:`n = |\Delta^{\mathcal{I}}|` canonical elements (satisfiable concept names in :math:`N_C \cup \{\top\}`)
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
   * - :class:`~mowl.geometric.init.ExactBoxInit`
     - Normalised :math:`\mathcal{EL}^{++}` (``method="projected"``) or propositional :math:`\mathcal{EL}_\bot` with fallback to ``"projected"`` (``method="milp"``)
     - In ``"projected"`` mode with :math:`k = 1`, guarantees :math:`\text{box}(C) \subseteq \text{box}(D)` (up to ``float32`` tolerance ``1e-6``) and zero ``gci0`` loss at margin :math:`0` for every entailed atomic subsumption :math:`C \sqsubseteq D` with **satisfiable** :math:`C` (unsatisfiable subclasses are points at the origin and are not covered; disjointness is not guaranteed). In ``"milp"`` mode, preserves both containment and disjointness when a box model is found (its boxes are mapped coordinate-wise by an increasing affine map from the solver's universe :math:`[0, M]` onto :math:`[-\text{scale}, \text{scale}]`), and falls back to ``"projected"`` on :class:`~mowl.geometric.errors.NotBoxRepresentable` or :class:`~mowl.geometric.errors.BoxModelNotFound`. Skips classes absent from the canonical model (``NaN`` rows left untouched by :meth:`~mowl.geometric.init.ExactBoxInit.apply_to_module` and ignored by :meth:`~mowl.geometric.init.ExactBoxInit.anchor_loss`).
     - Target :math:`(k, d)` matching :class:`~mowl.models.BoxSquaredEL`, :class:`~mowl.models.ELBE` (:math:`k = 1`), or :class:`~mowl.models.MultiBoxEL` (:math:`k \ge 1`)
   * - :class:`~mowl.models.MultiBoxEL` / :class:`~mowl.nn.MultiBoxELModule`
     - :math:`\mathcal{EL}^{++}` concept normal forms (``gci0``, ``gci1``, ``gci2``, ``gci3``, ``gci0_bot``, ``gci1_bot``, ``gci3_bot``) and ABox assertions (``role_axiom_capable = False``)
     - Every class is a union of :math:`k = \text{n\_components}` axis-parallel boxes with shared class bump and Box²EL role semantics. At :math:`k = 1` with ``component_reg=0.0``, every GCI loss (positive and negative, including ``gci1`` batches with empty intersections) and regularisation loss equal :class:`~mowl.models.BoxSquaredEL` to within ``1e-6``. In ``gci1``, a detached argmin over emptiness selects a designated pair :math:`(i^*, j^*)` and a detached gate :math:`g_{ij} = 1` iff :math:`\text{lo}_{ij} \le \text{up}_{ij}` in all coordinates or :math:`(i, j) = (i^*, j^*)` imposes inclusion only on non-empty (plus the designated) component intersections. In ``gci1_bot``, zero loss at :math:`\gamma = 0` requires :math:`\text{dist}(C_i, D_j) \ge 0` in *every* coordinate across all :math:`k^2` pairs (zero loss implies disjointness up to boundary contact, but disjointness does not imply zero loss).
     - :math:`k` boxes in :math:`\mathbb{R}^d` per class (:math:`k \ge 2` escapes the Helly number 2 obstruction on :math:`T_3`)
   * - :class:`~mowl.geometric.boxes.BoxModelBuilder`
     - Propositional :math:`\mathcal{EL}_\bot` only (:class:`~mowl.geometric.normal_forms.NF1`, :class:`~mowl.geometric.normal_forms.NF2`, and :math:`\bot` forms)
     - Runs :func:`~mowl.geometric.helly.helly_check` first and, in both ``faithful`` modes, raises :class:`~mowl.geometric.errors.NotBoxRepresentable` iff a non-degenerate certificate is found (a proof that no non-degenerate box model exists); searches via MILP (`scipy.optimize.milp`) and verifies any found :class:`~mowl.geometric.boxes.BoxModel` with :class:`~mowl.geometric.checker.ModelChecker` (a time-limited incumbent is post-checked and returned only if it passes). Raises :class:`~mowl.geometric.errors.BoxModelNotFound` when inconclusive.
     - Searches :math:`d \in \{1, 2, 4, \dots, d_{\max}\}` (default :math:`d_{\max} = 8`)
   * - :class:`~mowl.geometric.checker.ModelChecker`
     - ``"interval"``, ``"polyhedral"`` (full normalised :math:`\mathcal{EL}^{++}`), and ``"box"`` (propositional :math:`\mathcal{EL}_\bot`)
     - Exact decision procedure for axiom satisfaction, non-degeneracy, and strong faithfulness on concept names (combinatorial for intervals/boxes/simplex faces; HiGHS linear programming for general polyhedra and polyhedral role axioms). Polyhedral role axioms are checked only for simplex-vertex regions (Face rows and PairHull halves one-hot or zero); other regions raise :class:`~mowl.geometric.errors.UnsupportedRegion`.
     - Matches interpretation ambient dimension


Limitations and caveats
-----------------------

1. **Polyhedral dimension equals canonical-model size minus one (:math:`k = |\Delta^{\mathcal{I}}| - 1`)**
   In :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`, the ambient
   dimension is :math:`k = n - 1`, where :math:`n = |\text{cm.names}|` includes
   all satisfiable original concept names, ``owl:Thing``, and any auxiliary
   concept names introduced by :class:`~mowl.ontology.normalize.ELNormalizer`.
   This is an upper bound: a dimension linear in the number of concept names
   is unavoidable in the worst case for convex models that realise the
   relevant conjunctions (Gutiérrez-Basulto & Schockaert, KR 2018, Prop. 4),
   but :math:`|\Delta^{\mathcal{I}}| - 1` itself, which counts auxiliary names
   and :math:`\top`, need not be minimal. LP checks over role polyhedra scale
   with :math:`n`.
   For large ontologies, prefer :class:`~mowl.geometric.intervals.IntervalUnionModel`.

2. **Polyhedral regions are lower-dimensional simplex faces unless ``thicken=True``**
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

3. **Box builder is propositional-only, may be inconclusive, and produces zero-width boxes**

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

4. **``HellyReport.complete`` and conservative ``box_representable_nondegenerate``**

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

5. **Lossy bounding-box export**
   :func:`~mowl.geometric.export.to_bounding_boxes` replaces each simplex face of
   a :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel` by its elementwise
   coordinate minimum and maximum. Because axis-parallel bounding boxes of
   simplex faces intersect strictly more often than the faces themselves, the
   resulting boxes **do not satisfy the ontology's disjointness or conjunction
   axioms** and must never be treated as a model.

6. **Projected box initialisation (:class:`~mowl.geometric.init.ExactBoxInit`) and ``certificate_triangles``**

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

7. **MultiBoxEL (:class:`~mowl.models.MultiBoxEL`, :class:`~mowl.nn.MultiBoxELModule`)**

   * *Exact* :math:`k = 1` *identity:* When ``n_components=1`` and
     ``component_reg=0.0``, every GCI loss (positive and negative, including
     ``gci1`` batches containing empty pairwise intersections) and
     :meth:`~mowl.nn.MultiBoxELModule.regularization_loss` agree with
     :class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule` to within
     ``1e-6``.
   * *Gated pairwise intersections in* ``gci1``: For :math:`C \sqcap D \sqsubseteq E`,
     let :math:`(i^*, j^*) = \arg\min_{i,j} \text{empt}(C_i, D_j)` be the
     designated component pair (selected without gradient). The detached gate
     :math:`g_{ij} = 1` if :math:`\text{lo}_{ij} \le \text{up}_{ij}` in every
     coordinate or :math:`(i, j) = (i^*, j^*)`, else :math:`0`, enforces
     inclusion into some component :math:`E_m` only on intersecting pairs (plus
     the designated pair, which also receives the non-emptiness penalty
     :math:`\text{empt}(C_{i^*}, D_{j^*})`), while disjoint non-designated pairs
     impose no penalty.
   * *Coordinatewise separation in* ``gci1_bot``: Inherited from Box²EL,
     ``gci1_bot_loss`` for :math:`C \sqcap D \sqsubseteq \bot` sums
     :math:`\text{relu}(-\text{dist}(C_i, D_j) - \gamma)` across all coordinates
     and all :math:`k^2` pairs. At :math:`\gamma = 0`, zero loss implies that
     all component pairs are disjoint (up to boundary contact), **but not
     conversely**: two boxes separated in one coordinate whose projections
     overlap in another coordinate are disjoint in :math:`\mathbb{R}^d` yet
     incur positive ``gci1_bot_loss``.
   * *Non-convexity:* Taking minima over target components and gating over
     pairwise intersections yields a piecewise-smooth non-convex objective;
     gradient-based training is not guaranteed to converge to a global optimum
     from arbitrary random initialisations.


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
