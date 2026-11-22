"""The representation of a method, split into parts.

:mod:`mowl.representation.vocabulary` names the parts (view, space, class model, relation
model, individual model, constructors, measure, parametrisation);
:mod:`mowl.representation.catalog` states them for every catalogued method.
"""

from mowl.representation.vocabulary import (
    View, Space, ClassModel, RelationModel, IndividualModel, Construction, Measure,
    Parametrisation, Composable, Representation, CONSTRUCTORS)
from mowl.representation.catalog import REPRESENTATIONS
from mowl.representation.regions import Region, Ball, Box, ConstructorUnavailable
from mowl.representation.measures import RegionMeasure, CentreDistance, Bounds
from mowl.representation.class_models import Classes, BallClasses, BoxClasses
from mowl.representation.relation_models import Relations, Translation
from mowl.representation.objectives import Objective, FORMS
from mowl.representation.geometry import Geometry, ComposedGeometry
from mowl.representation.methods import (METHODS, CLASS_MODELS, RELATION_MODELS, MEASURES,
                                         compose)
from mowl.representation.adapters import geometry_of, register_adapter, ADAPTERS

__all__ = ["View", "Space", "ClassModel", "RelationModel", "IndividualModel",
           "Construction", "Measure", "Parametrisation", "Composable", "Representation",
           "CONSTRUCTORS", "REPRESENTATIONS",
           "Region", "Ball", "Box", "ConstructorUnavailable",
           "RegionMeasure", "CentreDistance", "Bounds",
           "Classes", "BallClasses", "BoxClasses", "Relations", "Translation",
           "Objective", "FORMS", "Geometry", "ComposedGeometry",
           "geometry_of", "register_adapter", "ADAPTERS",
           "METHODS", "CLASS_MODELS", "RELATION_MODELS", "MEASURES", "compose"]
