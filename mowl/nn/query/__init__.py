"""Query embedding geometries (Query2Box, BetaE, ConE, AConE) and the shared operator tree of
class expressions (:mod:`mowl.nn.query.expression`).

The geometries are combinations of parts (:mod:`mowl.representation.library.query_methods`);
they are imported on first use."""
from mowl.nn.query.expression import (Node, Vocabulary, class_expression_to_query,  # noqa: F401
                                      from_nested, to_dnf)
from mowl.nn.query.base import QueryGeometry, UnsupportedOperator  # noqa: F401

_GEOMETRIES = ("Query2BoxGeometry", "BetaEGeometry", "ConEGeometry", "AConEGeometry")


def __getattr__(name):
    if name in _GEOMETRIES:
        from mowl.nn.query import geometry
        return getattr(geometry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
