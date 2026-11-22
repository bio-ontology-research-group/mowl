"""The constructed methods as named combinations of a source and a reading.

=================  ================  ==============  ==========================================
catalogue key      source            reading         options
=================  ================  ==============  ==========================================
canonical_model    canonical model   element sets
least_model        least model       element sets
interval_union     canonical model   intervals       half_width=0.25
convex_polyhedral  canonical model   simplex         thicken=False
convex_regions     finite model      simplex
alcone-exact       canonical model   half-axes       source: nominals=False
temporal_cones     lasso model       half-axes       reading: merge=True (one cell per trace)
attributed_dl      attributed        simplex
box_milp           box milp          boxes
=================  ================  ==============  ==========================================

Every row is a :class:`~mowl.representation.construction.Recipe` registered with
:func:`~mowl.representation.register_method` (``trained_on="construction"``), so
``METHODS[key](data)`` builds the geometry, and :class:`~mowl.models.constructed.ConstructedModel`
builds it for a dataset. Any source can be combined with any reading that reads what it
builds: the canonical model read as cones, as intervals or as faces of a simplex; the
least model or a Boolean TBox read as intervals; a temporal model read as element sets.
Box models are read as boxes only, and finite models are not read as single boxes
(:mod:`~mowl.representation.library.construct_readings` says why).

The classes users had for these methods (:class:`~mowl.geometric.intervals.IntervalUnionModel`,
:class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`, ...) are built through these
recipes; their ``module`` is the geometry. :func:`~mowl.representation.geometry_of` of
them, and of the canonical model, the least model, finite interpretations and box models,
is that geometry.
"""

from mowl.representation.adapters import register_adapter
from mowl.representation.construction import (RECIPES, Recipe, Source, construct,
                                              register_recipe, register_source)

__all__ = ["recipe_geometry", "cone_geometry", "ConeInterpretationSource"]

for _recipe in (
        Recipe("canonical_model", "canonical model", "element sets"),
        Recipe("least_model", "least model", "element sets"),
        Recipe("interval_union", "canonical model", "intervals"),
        Recipe("convex_polyhedral", "canonical model", "simplex"),
        Recipe("convex_regions", "finite model", "simplex"),
        Recipe("alcone-exact", "canonical model", "half-axes", {"nominals": False}),
        Recipe("temporal_cones", "lasso model", "half-axes", reading_options={"merge": True}),
        Recipe("attributed_dl", "attributed", "simplex"),
        Recipe("box_milp", "box milp", "boxes")):
    register_recipe(_recipe)


def recipe_geometry(key, built, **names):
    """The geometry of method ``key`` for a model its source has already built."""
    recipe = RECIPES[key]
    return construct(None, recipe.source, recipe.reading,
                     source_options=recipe.source_options,
                     reading_options=recipe.reading_options, built=built, **names)


@register_source
class ConeInterpretationSource(Source):
    """An al-cone interpretation given by its cones (no construction): read by the
    ``half-axes`` reading with one cell per half-axis."""

    key = "al-cone interpretation"
    kind = "given"
    logic = "ALC"
    reasoner = "none (the cones are given)"

    @property
    def produces(self):
        from mowl.geometric.cones import ALConeInterpretation
        return ALConeInterpretation

    def rejects(self, data):
        from mowl.geometric.cones import ALConeInterpretation
        if isinstance(data, ALConeInterpretation):
            return None
        return f"it reads an ALConeInterpretation, not a {type(data).__name__}"

    def build(self, data):
        return data

    def default_names(self, built):
        return sorted(built.concepts), sorted(built.roles), sorted(built.individuals)


def cone_geometry(interpretation):
    """The geometry of an al-cone interpretation given by its cones."""
    return construct(interpretation, "al-cone interpretation", "half-axes")


@register_adapter("IntervalUnionModel", "ConvexPolyhedralModel", "ConvexRegionModel",
                  "ALConeInterpretation", "TemporalALConeInterpretation",
                  "AttributedConvexModel")
def _module(model):
    return model.module


@register_adapter("CanonicalModel")
def _canonical(cm):
    return RECIPES["canonical_model"].build(cm)


@register_adapter("LeastModel")
def _least(lm):
    return RECIPES["least_model"].build(lm)


@register_adapter("FiniteInterpretation", "FiniteModel")
def _finite(model):
    return RECIPES["canonical_model"].build(model, source="finite model")


@register_adapter("BoxModel")
def _boxes(model):
    return RECIPES["box_milp"].build(model)
