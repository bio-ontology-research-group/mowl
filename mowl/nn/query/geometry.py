"""The query embedding geometries as modules: Query2Box, BetaE, ConE and AConE.

Each is a :class:`~mowl.representation.library.query_methods.ComposedQueryGeometry`: a
combination of an entity model, a relation model, an intersection operator, a negation
operator and a distance (:mod:`mowl.representation.library.query_parts`). The classes are
defined next to the builders of the methods, in
:mod:`mowl.representation.library.query_methods`, and re-exported here with their
constructors of old (``Query2BoxGeometry(num_entities, num_relations, dim, gamma, ...)``).
"""

from mowl.nn.query.base import QueryGeometry, UnsupportedOperator  # noqa: F401
from mowl.nn.query.expression import from_signature  # noqa: F401  (re-export convenience)
from mowl.representation.library.query_methods import (  # noqa: F401
    AConEGeometry, BetaEGeometry, ConEGeometry, Query2BoxGeometry)

__all__ = ["QueryGeometry", "Query2BoxGeometry", "BetaEGeometry", "ConEGeometry",
           "AConEGeometry", "UnsupportedOperator"]
