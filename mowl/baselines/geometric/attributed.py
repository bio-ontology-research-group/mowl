"""The direct construction of convex models of attributed ontologies (Bourgaux, Ozaki &
Pan, DL 2021, Theorems 6 and 12), kept for comparison (see :mod:`mowl.baselines`): the
grounding, the canonical model and the direct convex polyhedral model
(:mod:`mowl.baselines.geometric.polyhedral`) in one class. In
:mod:`mowl.baselines.geometric.core.attributed` it is the combination of the source ``attributed`` and the
``simplex`` reading.
"""
from typing import List

import numpy as np

from mowl.baselines.geometric.polyhedral import ConvexPolyhedralModel
from mowl.baselines.geometric.core.attributed import (ANY, AttributedAxiom, AttributedOntology, Specifier,
                                       _name, ground)
from mowl.baselines.geometric.core.completion import CanonicalModel
from mowl.baselines.geometric.core.normal_forms import NormalisedOntology, nominal_name

__all__ = ["AttributedConvexModel"]


class AttributedConvexModel:
    """A convex geometric model of a ground attributed ontology (Theorem 6 / 12).

    :param ontology: An :class:`AttributedOntology`.
    :raises mowl.baselines.geometric.core.errors.InconsistentOntology: if the ontology is unsatisfiable.
    :raises mowl.baselines.geometric.core.errors.UnsupportedAxiom: for ``before``/``after``/``between``,
        temporal specifiers on roles or in open specifiers.

    After construction: :attr:`grounding` (``dl(O_g)``), :attr:`canonical_model`
    (the finite model), :attr:`model` (the
    :class:`~mowl.baselines.geometric.core.polyhedral.ConvexPolyhedralModel` :math:`\\eta'`) and
    :attr:`dimension`.
    """

    #: The model is an exact construction, not trained.
    axiom_scoring = None

    def __init__(self, ontology: AttributedOntology):
        self.ontology = ontology
        self.grounding = ground(ontology)
        normalised = NormalisedOntology(
            list(self.grounding.axioms), set(self.grounding.concept_names),
            set(self.grounding.role_names), 0)
        self.canonical_model = CanonicalModel.from_normalised(normalised)
        self.model = ConvexPolyhedralModel(self.canonical_model)

    @property
    def dimension(self) -> int:
        return self.model.dimension

    # -- regions ------------------------------------------------------------
    def concept_region(self, concept: str, spec: Specifier = ANY):
        """:math:`\\eta(A@S)`: a face of the simplex (empty for specifiers that do not
        occur in the ontology)."""
        return self.model.class_region(_name(concept, spec))

    def role_region(self, role: str, spec: Specifier = ANY):
        """:math:`\\eta(R@S)`: convex hull of concatenated vertex pairs."""
        return self.model.role_region(_name(role, spec))

    def point(self, individual: str) -> np.ndarray:
        """:math:`\\eta(a)`: the simplex vertex of the nominal of ``a``."""
        cm = self.canonical_model
        return self.model.element_points()[cm.element_of(nominal_name(individual))]

    # -- what holds ---------------------------------------------------------
    def holds_concept(self, individual: str, concept: str, spec: Specifier = ANY) -> bool:
        """``True`` iff :math:`\\eta(a) \\in \\eta(A@S)`, i.e. :math:`\\eta \\models A(a)@S`."""
        region = self.concept_region(concept, spec)
        if region.is_empty:
            return False
        try:
            return bool(region.contains(self.point(individual)))
        except KeyError:
            return False

    def holds_role(self, subject: str, role: str, obj: str, spec: Specifier = ANY) -> bool:
        """``True`` iff :math:`\\eta(a) \\oplus \\eta(b) \\in \\eta(R@S)` (concatenation as the
        linear map ``f``). The region is the hull of vertex pairs, whose vertices are exactly
        the pairs of the canonical model, so membership of a vertex pair is membership of the
        pair in the canonical role."""
        cm = self.canonical_model
        try:
            i = cm.element_of(nominal_name(subject))
            j = cm.element_of(nominal_name(obj))
        except KeyError:
            return False
        return (i, j) in cm.role(_name(role, spec))

    def included(self, sub: str, sub_spec: Specifier, sup: str, sup_spec: Specifier = ANY) -> bool:
        """``True`` iff :math:`\\eta(A@S) \\subseteq \\eta(B@T)` for two concept names."""
        a, b = self.concept_region(sub, sub_spec), self.concept_region(sup, sup_spec)
        return a.is_empty or a.is_subset(b)

    def entails_instance(self, individual: str, concept: str, spec: Specifier = ANY) -> bool:
        """Entailment of ``A(a)@S`` by the grounded ontology (canonical model). Strongly
        faithful (:math:`\\varphi(\\eta) = M`): equals :meth:`holds_concept`."""
        return self.canonical_model.instance_of(individual, _name(concept, spec))

    def is_satisfiable(self, concept: str, spec: Specifier = ANY) -> bool:
        """``True`` iff ``A@S`` is not forced to be empty."""
        return self.canonical_model.is_satisfiable(_name(concept, spec))

    def satisfies(self, index: int) -> bool:
        """Whether the geometry satisfies the original axiom number ``index`` of the
        ontology (checked on its normal-form translation with the exact model checker)."""
        return all(self.model.satisfies(nf) for nf in self.grounding.per_axiom[index])

    def violations(self) -> List[AttributedAxiom]:
        """Original axioms that are not satisfied by the geometry (empty by construction)."""
        return [ax for k, ax in enumerate(self.ontology.axioms) if not self.satisfies(k)]

    def is_model(self) -> bool:
        """``True`` iff every original axiom and every specifier axiom ``E_S ⊑ E_T`` holds
        in the geometry."""
        return not self.violations() and self.model.is_model(self.grounding.axioms)
