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
:class:`ELBoxEmbeddings <mowl.models.elboxembeddings.model.ELBoxEmbeddings>`, and
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
  Guimarães, TGDK 2024, §5), and the dimension bound :math:`|\Delta^{\mathcal{I}}| - 1`
  cannot in general be reduced (Gutiérrez-Basulto & Schockaert, KR 2018, Prop. 4).
* **Unions of closed intervals on the real line** model every consistent
  normalised :math:`\mathcal{EL}^{++}` ontology *exactly* in one dimension, with
  roles interpreted as finite unions of 2-D axis-aligned rectangles (Lacerda,
  Ozaki & Guimarães, TGDK 2024, Def. 12; the :math:`n = 1` case of the multibox
  construction of Federico, Leemhuis & Kutz, JOWO 2026, Thm. 3).

The :mod:`mowl.geometric` package implements these constructive models, an exact
geometric model checker (:class:`~mowl.geometric.checker.ModelChecker`), a
Helly-satisfiability diagnostic with non-box-representability certificates
(:func:`~mowl.geometric.helly.helly_check`), and a mixed-integer linear
programming builder for axis-parallel box models
(:class:`~mowl.geometric.boxes.BoxModelBuilder`).


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
     - :math:`k = n - 1` (tight in general; `Gutiérrez-Basulto & Schockaert, KR 2018, Prop. 4`)
   * - :class:`~mowl.geometric.intervals.IntervalUnionModel`
     - Full normalised :math:`\mathcal{EL}^{++}`
     - Exact model (:math:`\eta \models \mathcal{O}`), non-degenerate, and strongly faithful on concept names (`Lacerda et al., TGDK 2024, Def. 12`; `Federico et al., JOWO 2026, Thm. 3`).
     - :math:`1` (classes in :math:`\mathbb{R}`, roles in :math:`\mathbb{R}^2`)
   * - :func:`~mowl.geometric.helly.helly_check`
     - Propositional :math:`\mathcal{EL}_\bot` (exact for strong reading when ``complete=True``); sound necessary condition with roles
     - Every :class:`~mowl.geometric.helly.HellyCertificate` is a sound proof that no box model realises its three pairwise conjunctions simultaneously (`Eckhoff, 1988`; `Leemhuis & Kutz, NeSy 2025, Props. 4, 5`).
     - Dimension-independent (excludes box models in *any* :math:`\mathbb{R}^d`)
   * - :class:`~mowl.geometric.boxes.BoxModelBuilder`
     - Propositional :math:`\mathcal{EL}_\bot` only (:class:`~mowl.geometric.normal_forms.NF1`, :class:`~mowl.geometric.normal_forms.NF2`, and :math:`\bot` forms)
     - Runs :func:`~mowl.geometric.helly.helly_check` first (raises :class:`~mowl.geometric.errors.NotBoxRepresentable` on certificate); searches via MILP (`scipy.optimize.milp`) and verifies any found :class:`~mowl.geometric.boxes.BoxModel` with :class:`~mowl.geometric.checker.ModelChecker`. Raises :class:`~mowl.geometric.errors.BoxModelNotFound` when inconclusive.
     - Searches :math:`d \in \{1, 2, 4, \dots, d_{\max}\}` (default :math:`d_{\max} = 8`)
   * - :class:`~mowl.geometric.checker.ModelChecker`
     - ``"interval"``, ``"polyhedral"`` (full normalised :math:`\mathcal{EL}^{++}`), and ``"box"`` (propositional :math:`\mathcal{EL}_\bot`)
     - Exact decision procedure for axiom satisfaction, non-degeneracy, and strong faithfulness on concept names (combinatorial for intervals/boxes/simplex faces; HiGHS linear programming for general polyhedra and polyhedral role axioms).
     - Matches interpretation ambient dimension


Limitations and caveats
-----------------------

1. **Polyhedral dimension equals canonical-model size minus one (:math:`k = |\Delta^{\mathcal{I}}| - 1`)**
   In :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`, the ambient
   dimension is :math:`k = n - 1`, where :math:`n = |\text{cm.names}|` includes
   all satisfiable original concept names, ``owl:Thing``, and any auxiliary
   concept names introduced by :class:`~mowl.ontology.normalize.ELNormalizer`.
   This dimension cannot in general be reduced (Gutiérrez-Basulto & Schockaert,
   KR 2018, Prop. 4), and LP checks over role polyhedra scale with :math:`n`.
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
   * *Inconclusive MILP outcome:* A passing Helly check does *not* guarantee that
     an axis-parallel box model exists (no such sufficiency theorem is known).
     Only a Helly certificate (:class:`~mowl.geometric.errors.NotBoxRepresentable`)
     is a proof of impossibility; when no certificate is found and the MILP fails
     for all :math:`d \le d_{\max}` or reaches ``time_limit``,
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

5. **Lossy bounding-box export**
   :func:`~mowl.geometric.export.to_bounding_boxes` replaces each simplex face of
   a :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel` by its elementwise
   coordinate minimum and maximum. Because axis-parallel bounding boxes of
   simplex faces intersect strictly more often than the faces themselves, the
   resulting boxes **do not satisfy the ontology's disjointness or conjunction
   axioms** and must never be treated as a model.


References
----------

* V. Gutiérrez-Basulto and S. Schockaert, *From Knowledge Graph Embedding to
  Ontology Embedding? An Analysis of the Compatibility between Vector Space
  Representations and Rules*, in **Proceedings of KR 2018** (arXiv:1805.10461),
  Propositions 3 and 4.
* T. Lacerda, A. Ozaki, and R. Guimarães, *Geometric Models for Expressive
  Description Logics*, **Transactions on Graph Data and Knowledge (TGDK)**
  2(3):2, 2024, Definitions 2, 8, 12 and Theorems 13, 16, 18, 25.
* M. Leemhuis and O. Kutz, *Wait, That's Not a Box! On the Box-Representability
  of Description Logic Ontologies*, in **Proceedings of NeSy 2025** (PMLR
  284:303–321), Definitions 1, 3, 6 and Propositions 4, 5, 8.
* M. Leemhuis and O. Kutz, **NAI 2026**, Proposition 38 (incompleteness of
  ABox-closure Helly tests in the presence of roles).
* I. Federico, M. Leemhuis, and O. Kutz, *Multibox Embeddings for ELHO(o)_bot*,
  in **Proceedings of JOWO 2026**, Theorem 3 (1-D case).
* C. Bourgaux, R. Guimarães, R. Koudijs, T. Lacerda, and A. Ozaki, in
  **Proceedings of KR 2024**, Definitions 5–6 and Remark 2.
* Y. Kazakov, M. Krötzsch, and F. Simančík, *The Incredible ELK: From Polynomial
  Procedures to Efficient Reasoning with EL Ontologies*, **Journal of Automated
  Reasoning** 53(1):1–61, 2014.
* F. Baader, S. Brandt, and C. Lutz, *Pushing the EL Envelope*, in
  **Proceedings of IJCAI 2005**.
* J. Eckhoff, *Intersection properties of boxes. Part I: An upper-bound
  theorem*, **Israel Journal of Mathematics** 62(3):283–301, 1988.


.. toctree::
   :maxdepth: 2

   examples
   api
