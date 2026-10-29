"""1-D interval-union exact geometric model for normalised EL++ ontologies.

Implements the constructive 1-D interval-union interpretation of Lacerda, Ozaki
& Guimarães (TGDK 2024, Definition 12) and the 1-D case (:math:`n = 1`) of the
multibox construction of Federico, Leemhuis & Kutz (JOWO 2026, Theorem 3).

Construction:
- Given a :class:`~mowl.geometric.completion.CanonicalModel` with domain
  :math:`\\Delta = \\{d_0, \\dots, d_{m-1}\\}` (indexed :math:`0 \\le i < m`),
  each canonical element at position :math:`i` maps to the closed interval
  :math:`I(i) = [i + 1 - 0.25,\\; i + 1 + 0.25] = [(i+1) - \\tfrac{1}{4},\\; (i+1) + \\tfrac{1}{4}]`.
- Universe :math:`U = \\eta(\\top) = \\bigcup_{0 \\le i < m} I(i)`.
- For every concept name :math:`A`,
  :math:`\\eta(A) = \\bigcup_{i \\in A^{\\mathcal{I}}} I(i)`
  (and :math:`\\eta(\\bot) = \\emptyset`, unknown names :math:`\\mapsto \\emptyset`).
- For every role name :math:`r`,
  :math:`\\eta(r) = \\bigcup_{(i, j) \\in r^{\\mathcal{I}}} I(i) \\times I(j)`
  (and unknown roles :math:`\\mapsto \\emptyset`).

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
  concept name :math:`A` in ``cm.names``.
- **Strongly faithful on names**: for all concept names :math:`A, B` in
  ``cm.names``,
  :math:`\\eta(A) \\subseteq \\eta(B) \\iff \\mathcal{O} \\models A \\sqsubseteq B`.
- :class:`~mowl.geometric.errors.InconsistentOntology` propagates from
  :class:`~mowl.geometric.completion.CanonicalModel` when the ontology is
  inconsistent.

References:
- V. Lacerda, A. Ozaki & R. Guimarães, *Strong Faithfulness for ELH Ontology
  Embeddings*, Transactions on Graph Data and Knowledge (TGDK) 2(3):2, 2024,
  Definitions 2, 8, 12, Theorems 13, 16, 18, 25.
- G. Federico, M. Leemhuis & O. Kutz, *Introducing Multiboxes: Increasing the
  Trustworthiness of Ontology Embeddings*, JOWO 2026, Theorem 3 (1-D case).
- V. Gutiérrez-Basulto & S. Schockaert, *From Knowledge Graph Embedding to
  Ontology Embedding? An Analysis of the Compatibility between Vector Space
  Representations and Rules*, KR 2018 (arXiv:1805.10461), Props. 3, 4.
- Kazakov, Krötzsch & Simančík, JAR 2014 (The Incredible ELK).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

from mowl.geometric.checker import GeometricInterpretation, ModelChecker
from mowl.geometric.errors import InternalConsistencyError
from mowl.geometric.normal_forms import BOTTOM, TOP, Axiom
from mowl.geometric.regions import EMPTY_1D, IntervalUnion, RectUnion

if TYPE_CHECKING:  # pragma: no cover
    from org.semanticweb.owlapi.model import OWLOntology
    from mowl.datasets import Dataset
    from mowl.geometric.completion import CanonicalModel

_HALF_WIDTH: float = 0.25


class IntervalUnionModel(GeometricInterpretation):
    """Exact 1-D interval-union model of a normalised EL++ ontology.

    Implements the construction of Lacerda, Ozaki & Guimarães (TGDK 2024,
    Def. 12) and the 1-D case of Federico, Leemhuis & Kutz (JOWO 2026, Thm. 3).

    Guarantees:
    - Exact model of the normalised ontology (``cm.axioms``), verified on
      construction via :class:`~mowl.geometric.checker.ModelChecker`.
    - Non-degenerate on all satisfiable concept names in ``cm.names``.
    - Strongly faithful on concept names in ``cm.names``.
    """

    family: str = "interval"

    def __init__(self, cm: CanonicalModel) -> None:
        self.canonical_model: CanonicalModel = cm
        self._names: List[str] = list(cm.names)
        n_elems = len(cm.domain)

        self._element_intervals: List[Tuple[float, float]] = [
            (float(i + 1) - _HALF_WIDTH, float(i + 1) + _HALF_WIDTH)
            for i in range(n_elems)
        ]
        self._universe: IntervalUnion = IntervalUnion(self._element_intervals)

        self._class_regions: Dict[str, IntervalUnion] = {}
        for cname in cm.concept_names:
            if cname == BOTTOM:
                self._class_regions[cname] = EMPTY_1D
            elif cname == TOP:
                self._class_regions[cname] = self._universe
            else:
                positions = cm.extension(cname)
                if not positions:
                    self._class_regions[cname] = EMPTY_1D
                else:
                    self._class_regions[cname] = IntervalUnion(
                        [self._element_intervals[i] for i in sorted(positions)]
                    )
        self._class_regions[BOTTOM] = EMPTY_1D
        self._class_regions[TOP] = self._universe

        self._role_regions: Dict[str, RectUnion] = {}
        for rname in cm.role_names:
            pairs = cm.role(rname)
            if not pairs:
                self._role_regions[rname] = RectUnion([])
            else:
                self._role_regions[rname] = RectUnion(
                    [
                        (self._element_intervals[i], self._element_intervals[j])
                        for (i, j) in sorted(pairs)
                    ]
                )

        self._checker: Optional[ModelChecker] = None
        checker = self.check()
        violated = checker.violations(cm.axioms)
        if violated or not checker.is_model(cm.axioms):
            raise InternalConsistencyError(
                f"IntervalUnionModel failed ModelChecker verification on "
                f"{len(violated)} axiom(s): {violated!r}"
            )

    @classmethod
    def from_canonical_model(cls, cm: CanonicalModel) -> IntervalUnionModel:
        """Build an :class:`IntervalUnionModel` from an existing :class:`~mowl.geometric.completion.CanonicalModel`."""
        return cls(cm)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> IntervalUnionModel:
        """Normalise a mOWL :class:`~mowl.datasets.Dataset`, saturate its canonical model, and build an :class:`IntervalUnionModel`."""
        from mowl.geometric.completion import CanonicalModel

        return cls(CanonicalModel.from_dataset(dataset))

    @classmethod
    def from_ontology(cls, ontology: OWLOntology) -> IntervalUnionModel:
        """Normalise an OWLAPI ``OWLOntology``, saturate its canonical model, and build an :class:`IntervalUnionModel`."""
        from mowl.geometric.completion import CanonicalModel

        return cls(CanonicalModel.from_ontology(ontology))

    @property
    def universe(self) -> IntervalUnion:
        """Return the universe region :math:`U = \\eta(\\top)` (union of all element intervals)."""
        return self._universe

    @property
    def names(self) -> Iterable[str]:
        """Return the satisfiable concept names in the canonical model's signature."""
        return list(self._names)

    @property
    def class_regions(self) -> Dict[str, IntervalUnion]:
        """Return a copy of the mapping from known concept names to :class:`~mowl.geometric.regions.IntervalUnion` regions."""
        return dict(self._class_regions)

    @property
    def role_regions(self) -> Dict[str, RectUnion]:
        """Return a copy of the mapping from known role names to :class:`~mowl.geometric.regions.RectUnion` regions."""
        return dict(self._role_regions)

    def element_interval(self, i: int) -> Tuple[float, float]:
        """Return the closed interval ``(i + 1 - 0.25, i + 1 + 0.25)`` for canonical element position ``i``."""
        if isinstance(i, bool) or not isinstance(i, int):
            raise TypeError(f"Element index must be an int, got {i!r}")
        if i < 0 or i >= len(self._element_intervals):
            raise IndexError(
                f"Element index {i} out of range [0, {len(self._element_intervals) - 1}]"
            )
        return self._element_intervals[i]

    def class_region(self, name: str) -> IntervalUnion:
        """Return :math:`\\eta(\\text{name})`, or :data:`~mowl.geometric.regions.EMPTY_1D` for ``BOTTOM`` and unknown names."""
        if name == BOTTOM:
            return EMPTY_1D
        if name == TOP:
            return self._universe
        return self._class_regions.get(name, EMPTY_1D)

    def role_region(self, role: str) -> RectUnion:
        """Return :math:`\\eta(\\text{role})`, or an empty :class:`~mowl.geometric.regions.RectUnion` for unknown roles."""
        return self._role_regions.get(role, RectUnion([]))

    def class_intervals(self, name: str) -> List[Tuple[float, float]]:
        """Return the list of ``(lo, hi)`` intervals forming :meth:`class_region` for ``name``."""
        return list(self.class_region(name).intervals)

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
