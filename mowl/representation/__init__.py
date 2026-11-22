"""The representation of a method, split into parts.

:mod:`mowl.representation.vocabulary` names the parts (view, space, class model, relation
model, individual model, constructors, measure, parametrisation);
:mod:`mowl.representation.catalog` states them for every catalogued method.
"""

import mowl.nn  # noqa: F401,E402  (first: the part-built modules of mowl.nn need this package complete)

from mowl.representation.vocabulary import (
    View, Space, ClassModel, RelationModel, IndividualModel, Construction, Measure,
    Parametrisation, Composable, Representation, CONSTRUCTORS)
from mowl.representation.catalog import REPRESENTATIONS
from mowl.representation.regions import (Region, Ball, Box, CornerBox,
                                         ConstructorUnavailable)
from mowl.representation.measures import RegionMeasure, CentreDistance, Bounds
from mowl.representation.class_models import Classes, BallClasses, BoxClasses
from mowl.representation.relation_models import Relations, Translation
from mowl.representation.objectives import Objective, FORMS, ROLE_FORMS
from mowl.representation.geometry import Geometry, ComposedGeometry
from mowl.representation.methods import (METHODS, METHOD_KINDS, CLASS_MODELS,
                                         RELATION_MODELS, MEASURES, DEFAULT_MEASURE,
                                         compose, register_class_model,
                                         register_relation_model, register_measure,
                                         register_method)
from mowl.representation.adapters import geometry_of, register_adapter, ADAPTERS

__all__ = ["View", "Space", "ClassModel", "RelationModel", "IndividualModel",
           "Construction", "Measure", "Parametrisation", "Composable", "Representation",
           "CONSTRUCTORS", "REPRESENTATIONS",
           "Region", "Ball", "Box", "CornerBox", "ConstructorUnavailable",
           "RegionMeasure", "CentreDistance", "Bounds",
           "Classes", "BallClasses", "BoxClasses", "Relations", "Translation",
           "Objective", "FORMS", "ROLE_FORMS", "Geometry", "ComposedGeometry",
           "geometry_of", "register_adapter", "ADAPTERS",
           "METHODS", "METHOD_KINDS", "CLASS_MODELS", "RELATION_MODELS", "MEASURES",
           "DEFAULT_MEASURE",
           "compose", "register_class_model", "register_relation_model",
           "register_measure", "register_method"]

from mowl.representation import library  # noqa: E402,F401  (registers further parts)
