"""The representation of a method, split into parts.

:mod:`mowl.representation.vocabulary` names the parts (view, space, class model, relation
model, individual model, constructors, measure, parametrisation);
:mod:`mowl.representation.catalog` states them for every catalogued method.
"""

from mowl.representation.vocabulary import (
    View, Space, ClassModel, RelationModel, IndividualModel, Construction, Measure,
    Parametrisation, Composable, Representation, CONSTRUCTORS)
from mowl.representation.catalog import REPRESENTATIONS

__all__ = ["View", "Space", "ClassModel", "RelationModel", "IndividualModel",
           "Construction", "Measure", "Parametrisation", "Composable", "Representation",
           "CONSTRUCTORS", "REPRESENTATIONS"]
