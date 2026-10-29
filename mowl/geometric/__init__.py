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
from mowl.geometric.regions import (
    EMPTY_1D,
    Box,
    Face,
    IntervalUnion,
    PairHull,
    RectUnion,
    simplex_points,
)

_CHECKER_NAMES = frozenset(
    {"DictInterpretation", "GeometricInterpretation", "ModelChecker"}
)
_COMPLETION_NAMES = frozenset({"CanonicalModel", "Completion", "CompletionResult"})
_NORMAL_FORMS_NAMES = frozenset(
    {
        "Axiom",
        "BOTTOM",
        "NF1",
        "NF2",
        "NF3",
        "NF4",
        "NF6",
        "NF7",
        "NormalisedOntology",
        "TOP",
        "normalise",
        "to_owl",
    }
)


def __getattr__(name: str) -> object:
    if name == "checker":
        import importlib

        return importlib.import_module("mowl.geometric.checker")
    if name == "completion":
        import importlib

        return importlib.import_module("mowl.geometric.completion")
    if name == "normal_forms":
        import importlib

        return importlib.import_module("mowl.geometric.normal_forms")
    if name in _CHECKER_NAMES:
        from mowl.geometric import checker

        return getattr(checker, name)
    if name in _COMPLETION_NAMES:
        from mowl.geometric import completion

        return getattr(completion, name)
    if name in _NORMAL_FORMS_NAMES:
        from mowl.geometric import normal_forms

        return getattr(normal_forms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Axiom",
    "BOTTOM",
    "Box",
    "BoxModelNotFound",
    "CanonicalModel",
    "Completion",
    "CompletionResult",
    "DictInterpretation",
    "EMPTY_1D",
    "Face",
    "GeometricError",
    "GeometricInterpretation",
    "InconsistentOntology",
    "InternalConsistencyError",
    "IntervalUnion",
    "MissingDependency",
    "ModelChecker",
    "NF1",
    "NF2",
    "NF3",
    "NF4",
    "NF6",
    "NF7",
    "NormalisedOntology",
    "NotBoxRepresentable",
    "PairHull",
    "RectUnion",
    "TOP",
    "UnsupportedAxiom",
    "UnsupportedRegion",
    "checker",
    "completion",
    "normal_forms",
    "normalise",
    "simplex_points",
    "to_owl",
]
