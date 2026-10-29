"""Convex polyhedral exact geometric model for normalised EL++ ontologies.

Implements the constructive convex polyhedral interpretation of
Gutiérrez-Basulto & Schockaert (KR 2018, Proposition 3) restated in standard
simplex coordinates (Lacerda, Ozaki & Guimarães, TGDK 2024, §5).

Construction:
- Given a :class:`~mowl.geometric.completion.CanonicalModel` with
  :math:`n = |\\text{cm.names}|` satisfiable concept names (including
  :math:`\\top`), let :math:`k = n - 1` be the ambient dimension and
  :math:`v_0, \\dots, v_{n-1} \\in \\mathbb{R}^k` be the rows of
  :func:`~mowl.geometric.regions.simplex_points` (row :math:`i` = element
  :math:`i`).
- Universe :math:`U = \\eta(\\top)` is the :class:`~mowl.geometric.regions.Face`
  spanned by all :math:`n` vertices :math:`\\{0, \\dots, n - 1\\}`.
- For every concept name :math:`A`, :math:`\\eta(A)` is the
  :class:`~mowl.geometric.regions.Face` with
  ``vertex_ids = cm.extension(A)`` (:math:`F(A^{\\mathcal{I}}) = \\text{conv}\\{v_i : i \\in A^{\\mathcal{I}}\\}`,
  and the empty :class:`~mowl.geometric.regions.Face` for ``BOTTOM`` or unknown
  names).
- For every role name :math:`r`, :math:`\\eta(r)` is the
  :class:`~mowl.geometric.regions.PairHull` whose rows are the concatenated
  vertex pairs :math:`v_i \\oplus v_j \\in \\mathbb{R}^{2k}` over
  :math:`(i, j) \\in r^{\\mathcal{I}}` (and an empty
  :class:`~mowl.geometric.regions.PairHull` of shape ``(0, 2k)`` for unknown
  or empty roles).
- Optional ``thicken=True`` (propositional fragment only, when ``cm.axioms``
  contains no :class:`~mowl.geometric.normal_forms.NF3`,
  :class:`~mowl.geometric.normal_forms.NF4`,
  :class:`~mowl.geometric.normal_forms.NF6`, or
  :class:`~mowl.geometric.normal_forms.NF7` axioms; otherwise raises
  :class:`~mowl.geometric.errors.UnsupportedAxiom`): replaces each class
  :class:`~mowl.geometric.regions.Face` by a
  :class:`~mowl.geometric.regions.ThickFace` with :math:`\\varepsilon = \\tfrac{1}{2n}`
  whose H-representation intersects the simplex constraints
  :math:`\\lambda_j(x) \\ge 0` with :math:`\\lambda_j(x) \\le \\varepsilon` for
  :math:`j \\notin S`, yielding full-dimensional regions with non-empty interior
  for satisfiable concepts.

Guarantees:
- **Exact model of the normalised ontology**: satisfies every normalised axiom
  (:class:`~mowl.geometric.normal_forms.NF1`–:class:`~mowl.geometric.normal_forms.NF4`,
  :class:`~mowl.geometric.normal_forms.NF6`,
  :class:`~mowl.geometric.normal_forms.NF7`, and :math:`\\bot` forms) of
  ``cm.axioms``, verified on construction by
  :class:`~mowl.geometric.checker.ModelChecker`. Raises
  :class:`~mowl.geometric.errors.InternalConsistencyError` listing any violated
  axioms if verification fails.
- **Non-degenerate**: :math:`\\eta(A) \\neq \\emptyset` for every satisfiable
  concept name :math:`A` in ``cm.names`` (regions are non-empty,
  lower-dimensional faces of the simplex by design when ``thicken=False``).
- **Strongly faithful on names**: for all concept names :math:`A, B` in
  ``cm.names``,
  :math:`\\eta(A) \\subseteq \\eta(B) \\iff \\mathcal{O} \\models A \\sqsubseteq B`.
- **Dimension**: ``dimension == len(cm.names) - 1`` (canonical-model size
  :math:`- 1`). This includes auxiliary names and :math:`\\top` and is an upper
  bound; a dimension linear in the number of concept names is unavoidable in
  the worst case for convex models that realise the relevant conjunctions
  (Gutiérrez-Basulto & Schockaert, KR 2018, Proposition 4).
- :class:`~mowl.geometric.errors.InconsistentOntology` propagates from
  :class:`~mowl.geometric.completion.CanonicalModel` when the ontology is
  inconsistent.

References:
- V. Gutiérrez-Basulto & S. Schockaert, *From Knowledge Graph Embedding to
  Ontology Embedding? An Analysis of the Compatibility between Vector Space
  Representations and Rules*, KR 2018 (arXiv:1805.10461), Propositions 3, 4.
- V. Lacerda, A. Ozaki & R. Guimarães, *Strong Faithfulness for ELH Ontology
  Embeddings*, TGDK 2(3):2, 2024, Definitions 2, 8, 12, Theorems 13, 16, 18, 25.
- M. Leemhuis & O. Kutz, *Understanding the Expressive Capabilities of
  Knowledge Base Embeddings under Box Semantics*, NeSy 2025 (PMLR
  284:303–321), Defs. 1, 3, 6, Props. 4, 5, 8.
- C. Bourgaux, R. Guimarães, R. Koudijs, V. Lacerda & A. Ozaki, *Knowledge
  Base Embeddings: Semantics and Theoretical Properties*, KR 2024, Defs. 5–6,
  Remark 2.
- Kazakov, Krötzsch & Simančík, JAR 2014 (The Incredible ELK).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional
import numpy as np

from mowl.geometric.checker import GeometricInterpretation, ModelChecker
from mowl.geometric.errors import InternalConsistencyError, UnsupportedAxiom
from mowl.geometric.normal_forms import (
    BOTTOM,
    NF3,
    NF4,
    NF6,
    NF7,
    TOP,
    Axiom,
)
from mowl.geometric.regions import Face, PairHull, ThickFace, simplex_points

if TYPE_CHECKING:  # pragma: no cover
    from org.semanticweb.owlapi.model import OWLOntology
    from mowl.datasets import Dataset
    from mowl.geometric.completion import CanonicalModel


class ConvexPolyhedralModel(GeometricInterpretation):
    """Exact convex polyhedral model of a normalised EL++ ontology.

    Implements the construction of Gutiérrez-Basulto & Schockaert (KR 2018,
    Proposition 3) in standard simplex coordinates in :math:`\\mathbb{R}^{n-1}`
    where :math:`n = |\\text{cm.names}|`.

    Guarantees:
    - Exact model of the normalised ontology (``cm.axioms``), verified on
      construction via :class:`~mowl.geometric.checker.ModelChecker`.
    - Non-degenerate (non-empty, lower-dimensional simplex faces) on all
      satisfiable concept names in ``cm.names``.
    - Strongly faithful on concept names in ``cm.names``.
    - ``dimension == len(cm.names) - 1`` (canonical-model size :math:`- 1`).
      This includes auxiliary names and :math:`\\top` and is an upper bound; a
      dimension linear in the number of concept names is unavoidable in the
      worst case for convex models that realise the relevant conjunctions
      (Gutiérrez-Basulto & Schockaert, KR 2018, Proposition 4).
    """

    family: str = "polyhedral"

    def __init__(self, cm: CanonicalModel, thicken: bool = False) -> None:
        if thicken and any(
            isinstance(ax, (NF3, NF4, NF6, NF7)) for ax in cm.axioms
        ):
            raise UnsupportedAxiom(
                "thicken=True is only supported for propositional ontologies "
                "(no NF3, NF4, NF6, or NF7 role axioms)"
            )

        self.canonical_model: CanonicalModel = cm
        self.thicken: bool = bool(thicken)
        self._names: List[str] = list(cm.names)
        n = len(self._names)
        k = n - 1
        self.dimension: int = k
        self._points: np.ndarray = simplex_points(n)
        eps = 1.0 / (2 * n)

        def _make_face(vertex_ids: Iterable[int]) -> Face:
            if self.thicken:
                return ThickFace.from_simplex(self._points, vertex_ids, eps=eps)
            return Face.from_simplex(self._points, vertex_ids)

        self._universe: Face = _make_face(range(n))
        self._empty_face: Face = _make_face(frozenset())
        self._empty_role: PairHull = PairHull(np.zeros((0, 2 * k), dtype=float))

        self._class_regions: Dict[str, Face] = {}
        for cname in cm.concept_names:
            if cname == BOTTOM:
                self._class_regions[cname] = self._empty_face
            elif cname == TOP:
                self._class_regions[cname] = self._universe
            else:
                positions = cm.extension(cname)
                if not positions:
                    self._class_regions[cname] = self._empty_face
                else:
                    self._class_regions[cname] = _make_face(positions)
        self._class_regions[BOTTOM] = self._empty_face
        self._class_regions[TOP] = self._universe

        self._role_regions: Dict[str, PairHull] = {}
        for rname in cm.role_names:
            pairs = cm.role(rname)
            if not pairs:
                self._role_regions[rname] = self._empty_role
            else:
                sorted_pairs = sorted(pairs)
                pair_rows = [
                    np.concatenate([self._points[i], self._points[j]])
                    for (i, j) in sorted_pairs
                ]
                pts_2k = np.vstack(pair_rows).reshape((len(sorted_pairs), 2 * k))
                self._role_regions[rname] = PairHull(pts_2k)

        self._checker: Optional[ModelChecker] = None
        checker = self.check()
        violated = checker.violations(cm.axioms)
        if violated or not checker.is_model(cm.axioms):
            raise InternalConsistencyError(
                f"ConvexPolyhedralModel failed ModelChecker verification on "
                f"{len(violated)} axiom(s): {violated!r}"
            )

    @classmethod
    def from_canonical_model(
        cls, cm: CanonicalModel, thicken: bool = False
    ) -> ConvexPolyhedralModel:
        """Build a :class:`ConvexPolyhedralModel` from an existing :class:`~mowl.geometric.completion.CanonicalModel`."""
        return cls(cm, thicken=thicken)

    @classmethod
    def from_dataset(
        cls, dataset: Dataset, thicken: bool = False
    ) -> ConvexPolyhedralModel:
        """Normalise a mOWL :class:`~mowl.datasets.Dataset`, saturate its canonical model, and build a :class:`ConvexPolyhedralModel`."""
        from mowl.geometric.completion import CanonicalModel

        return cls(CanonicalModel.from_dataset(dataset), thicken=thicken)

    @classmethod
    def from_ontology(
        cls, ontology: OWLOntology, thicken: bool = False
    ) -> ConvexPolyhedralModel:
        """Normalise an OWLAPI ``OWLOntology``, saturate its canonical model, and build a :class:`ConvexPolyhedralModel`."""
        from mowl.geometric.completion import CanonicalModel

        return cls(CanonicalModel.from_ontology(ontology), thicken=thicken)

    @property
    def universe(self) -> Face:
        """Return the universe region :math:`U = \\eta(\\top)` (the full simplex)."""
        return self._universe

    @property
    def names(self) -> Iterable[str]:
        """Return the satisfiable concept names in the canonical model's signature."""
        return list(self._names)

    @property
    def class_regions(self) -> Dict[str, Face]:
        """Return a copy of the mapping from known concept names to :class:`~mowl.geometric.regions.Face` regions."""
        return dict(self._class_regions)

    @property
    def role_regions(self) -> Dict[str, PairHull]:
        """Return a copy of the mapping from known role names to :class:`~mowl.geometric.regions.PairHull` regions."""
        return dict(self._role_regions)

    def element_points(self) -> np.ndarray:
        """Return the ``(n, n - 1)`` array of canonical-element simplex vertices (row ``i`` = element ``i``)."""
        return self._points.copy()

    def class_points(self, name: str) -> np.ndarray:
        """Return the ``(m, n - 1)`` array of vertex rows spanning :meth:`class_region` for ``name``."""
        return self.class_region(name).vertices.copy()

    def class_region(self, name: str) -> Face:
        """Return :math:`\\eta(\\text{name})`, or an empty :class:`~mowl.geometric.regions.Face` for ``BOTTOM`` and unknown names."""
        if name == BOTTOM:
            return self._empty_face
        if name == TOP:
            return self._universe
        return self._class_regions.get(name, self._empty_face)

    def role_region(self, role: str) -> PairHull:
        """Return :math:`\\eta(\\text{role})`, or an empty :class:`~mowl.geometric.regions.PairHull` for unknown roles."""
        return self._role_regions.get(role, self._empty_role)

    def check(self) -> ModelChecker:
        """Return the cached :class:`~mowl.geometric.checker.ModelChecker` bound to this interpretation."""
        if self._checker is None:
            self._checker = ModelChecker(self)
        return self._checker

    def satisfies(self, axiom: Axiom) -> bool:
        """Return ``True`` iff this model satisfies ``axiom`` (via :meth:`check`)."""
        return self.check().satisfies(axiom)

    def is_model(self, axioms: Optional[Iterable[Axiom]] = None) -> bool:
        """Return ``True`` iff this interpretation satisfies ``axioms`` (defaults to ``self.canonical_model.axioms``)."""
        target = self.canonical_model.axioms if axioms is None else axioms
        return self.check().is_model(target)

    def is_nondegenerate(self, cm: Optional[CanonicalModel] = None) -> bool:
        """Return ``True`` iff every satisfiable name in ``cm`` (defaults to ``self.canonical_model``) has a non-empty region."""
        target = self.canonical_model if cm is None else cm
        return self.check().is_nondegenerate(target)

    def is_strongly_faithful_on_names(
        self, cm: Optional[CanonicalModel] = None
    ) -> bool:
        """Return ``True`` iff region inclusion on names matches entailment in ``cm`` (defaults to ``self.canonical_model``)."""
        target = self.canonical_model if cm is None else cm
        return self.check().is_strongly_faithful_on_names(target)
