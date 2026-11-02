"""Exception hierarchy for constructive geometric models of EL++ ontologies.

Guarantees:
- All domain-specific exceptions raised by :mod:`mowl.geometric` inherit from
  :class:`GeometricError`.
- :class:`NotBoxRepresentable` carries a ``report`` attribute holding the
  :class:`~mowl.geometric.helly.HellyReport` certificate that proves no box
  model of the requested faithfulness exists (Leemhuis & Kutz, NeSy 2025,
  Defs. 1, 3, 6, Props. 4, 5, 8).
- :class:`BoxModelNotFound` represents an inconclusive MILP search outcome
  (distinct from :class:`NotBoxRepresentable`, which is a proof of impossibility).

References:
- Gutiérrez-Basulto & Schockaert, KR 2018 (arXiv:1805.10461), Props. 3, 4.
- Lacerda, Ozaki & Guimarães, TGDK 2(3):2, 2024, Defs. 2, 8, 12, Thms. 13, 16, 18, 25.
- Leemhuis & Kutz, NeSy 2025 (PMLR 284:303–321), Defs. 1, 3, 6, Props. 4, 5, 8.
- Bourgaux, Guimarães, Koudijs, Lacerda & Ozaki, KR 2024, Defs. 5–6, Remark 2.
- Kazakov, Krötzsch & Simančík, JAR 2014 (ELK).
"""
from __future__ import annotations

from typing import Any, Optional


class GeometricError(Exception):
    """Base exception for all errors in :mod:`mowl.geometric`."""


class InconsistentOntology(GeometricError):
    """Raised when an ontology entails ``owl:Thing <= owl:Nothing``."""


class UnsupportedAxiom(GeometricError):
    """Raised when an operation encounters an axiom outside its supported fragment."""


UnsupportedAxiomError = UnsupportedAxiom


class UnsupportedRegion(GeometricError):
    """Raised when a model checker or builder encounters an unsupported region representation."""


class NotBoxRepresentable(GeometricError):
    """Raised when a Helly certificate proves that no box model exists.

    Can be constructed either as ``NotBoxRepresentable(message, report)`` or as
    ``NotBoxRepresentable(report)``. The ``report`` attribute always stores the
    certificate report object.
    """

    def __init__(self, message_or_report: Any, report: Optional[Any] = None) -> None:
        if report is None and not isinstance(message_or_report, str):
            self.report = message_or_report
            message = f"Ontology is not box-representable: {message_or_report}"
        else:
            self.report = report
            message = str(message_or_report)
        super().__init__(message)


class BoxModelNotFound(GeometricError):
    """Raised when the MILP box-model search finishes without finding a model.

    This outcome is inconclusive: passing the Helly check does not guarantee
    that an axis-aligned box model exists within ``d_max`` dimensions or the
    configured solver time limit.
    """


class InternalConsistencyError(GeometricError):
    """Raised when a constructed geometric model fails verification by :class:`ModelChecker`."""


class MissingDependency(GeometricError):
    """Raised when an optional or version-gated dependency (e.g. ``scipy.optimize.milp``) is unavailable."""
