"""Constructive geometric models of EL++ ontologies.

Guarantees:
- Provides normal-form extraction (:func:`normalise`, :class:`NormalisedOntology`),
  ELK-style completion and canonical models, exact region models (convex
  polyhedral and 1-D interval unions), an exact geometric model checker,
  Helly-satisfiability diagnostics with non-representability certificates, and
  MILP-based axis-aligned box-model construction.

References:
- Gutiérrez-Basulto & Schockaert, KR 2018 (arXiv:1805.10461), Props. 3, 4.
- Lacerda, Ozaki & Guimarães, TGDK 2(3):2, 2024, Defs. 2, 8, 12, Thms. 13, 16, 18, 25.
- Leemhuis & Kutz, NeSy 2025 (PMLR 284:303–321), Defs. 1, 3, 6, Props. 4, 5, 8.
- Bourgaux, Guimarães, Koudijs, Lacerda & Ozaki, KR 2024, Defs. 5–6, Remark 2.
- Kazakov, Krötzsch & Simančík, JAR 2014 (The Incredible ELK).
"""
from __future__ import annotations

from mowl.geometric.completion import (
    CanonicalModel,
    Completion,
    CompletionResult,
)
from mowl.geometric.errors import (
    BoxModelNotFound,
    GeometricError,
    InconsistentOntology,
    InternalConsistencyError,
    MissingDependency,
    NotBoxRepresentable,
    UnsupportedAxiom,
    UnsupportedRegion,
)
from mowl.geometric.normal_forms import (
    BOTTOM,
    NF1,
    NF2,
    NF3,
    NF4,
    NF6,
    NF7,
    TOP,
    Axiom,
    NormalisedOntology,
    normalise,
    to_owl,
)

__all__ = [
    "Axiom",
    "BOTTOM",
    "BoxModelNotFound",
    "CanonicalModel",
    "Completion",
    "CompletionResult",
    "GeometricError",
    "InconsistentOntology",
    "InternalConsistencyError",
    "MissingDependency",
    "NF1",
    "NF2",
    "NF3",
    "NF4",
    "NF6",
    "NF7",
    "NormalisedOntology",
    "NotBoxRepresentable",
    "TOP",
    "UnsupportedAxiom",
    "UnsupportedRegion",
    "normalise",
    "to_owl",
]
