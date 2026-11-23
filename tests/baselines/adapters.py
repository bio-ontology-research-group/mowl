"""Reading a baseline object as a library geometry: the conversions of the tests.

The library does not know :mod:`mowl.baselines` exists. Some tests describe a direct
implementation by the library's geometry of the same method, built on the baseline's own
tables (the builders' ``tables=`` option), to check that the two read the same parameters
the same way. The adapters that do this live here, registered in the library's registry
of adapters (:func:`mowl.representation.adapters.register_adapter`) under the names of the
baseline classes, and only when the tests import this module (``tests/conftest.py`` does).

Two conversions are needed besides the adapters:

* :func:`library_manifold`: a baseline keeps its own copy of the manifolds; the library's
  measures check ``isinstance`` against the library classes, so a geometry on the
  baseline's tables takes the library manifold of the same name and attributes.
* :func:`library_reading`: a baseline constructed model checks normal forms of its own copy
  of the geometry code; the library asks with its own normal forms, which are converted.
"""
import dataclasses
import importlib
import os
import re

import mowl

from mowl.representation import methods
from mowl.representation.adapters import ADAPTERS, register_adapter


# -- conversions ---------------------------------------------------------------------------
def library_manifold(manifold):
    """The library manifold with the class name and attributes of a baseline manifold."""
    from mowl.representation.library import manifolds
    if manifold is None or type(manifold).__module__ == manifolds.__name__:
        return manifold
    cls = getattr(manifolds, type(manifold).__name__, None)
    if not (isinstance(cls, type) and issubclass(cls, manifolds.Manifold)):
        return manifold
    copy = cls.__new__(cls)
    copy.__dict__.update(vars(manifold))
    return copy


class _WithLibraryManifold:
    """The tables of a baseline module, with its manifold read as the library's."""

    def __init__(self, module):
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "manifold", library_manifold(module.manifold))

    def __getattr__(self, name):
        return getattr(self._module, name)


class _BaselineReading:
    """A baseline constructed model that answers ``satisfies`` for the library's normal
    forms (converted to the baseline's own normal-form classes) and errors (the
    baseline's ``UnsupportedAxiom``/``UnsupportedRegion`` raised as the library's)."""

    def __init__(self, model):
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def satisfies(self, axiom):
        from mowl.geometric import errors as lib_errors
        nf = importlib.import_module("mowl.baselines.geometric.core.normal_forms")
        errors = importlib.import_module("mowl.baselines.geometric.core.errors")
        if dataclasses.is_dataclass(axiom) and hasattr(nf, type(axiom).__name__):
            axiom = getattr(nf, type(axiom).__name__)(
                *[getattr(axiom, f.name) for f in dataclasses.fields(axiom)])
        try:
            return self._model.satisfies(axiom)
        except errors.UnsupportedAxiom as error:
            raise lib_errors.UnsupportedAxiom(str(error)) from error
        except errors.UnsupportedRegion as error:
            raise lib_errors.UnsupportedRegion(str(error)) from error


def library_reading(model):
    """``model`` as the library's constructed geometries read it (see :class:`_BaselineReading`):
    the baseline's model with the library's normal forms."""
    if not type(model).__module__.startswith("mowl.baselines"):
        return model
    return _BaselineReading(model)


class _FaithELReading:
    """A baseline FaithEL whose TBox model answers the library's normal forms."""

    def __init__(self, model):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "tbox", library_reading(model.tbox))

    def __getattr__(self, name):
        return getattr(self._model, name)


def faithel_reading(model):
    """A baseline FaithEL as :class:`~mowl.representation.library.other_regions.FaithELGeometry`
    reads it (its TBox model through :func:`library_reading`)."""
    return _FaithELReading(model)


# -- the adapters of the baseline modules ---------------------------------------------------
def _lib(name):
    return importlib.import_module(f"mowl.representation.library.{name}")


_BASELINES = os.path.join(os.path.dirname(mowl.__file__), "baselines")
_CLASS_PATHS = {}


def baseline_classes(name):
    """The qualified names of the classes called ``name`` that the baselines define (found by
    reading their sources, so that nothing is imported here)."""
    if not _CLASS_PATHS:
        pattern = re.compile(r"^class (\w+)", re.M)
        for directory, _, files in os.walk(_BASELINES):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(directory, file)
                    module = os.path.relpath(path[:-3], os.path.dirname(_BASELINES)).replace(
                        os.sep, ".")
                    module = "mowl." + module if not module.startswith("mowl.") else module
                    if module.endswith(".__init__"):
                        module = module[:-9]
                    for cls in pattern.findall(open(path).read()):
                        _CLASS_PATHS.setdefault(cls, []).append(f"{module}.{cls}")
    return _CLASS_PATHS.get(name, [])


def _adapter(*names):
    """Register an adapter for the baseline classes called ``names`` (the library's classes of
    the same name have their own adapters, or are geometries themselves)."""
    def decorate(function):
        paths = [path for name in names for path in baseline_classes(name)]
        assert paths, names
        return register_adapter(*paths)(function)
    return decorate


def mirror_library_adapters():
    """The baseline class of the same name as a library class with an adapter is read by that
    adapter (the direct implementation of the model of FaithEL, of the word2vec and graph
    models, of the constructed models): the pairs are two implementations of one method."""
    for qualified in list(ADAPTERS):
        function = ADAPTERS[qualified]
        if qualified.startswith("mowl.baselines."):
            continue
        for path in baseline_classes(qualified.rsplit(".", 1)[-1]):
            ADAPTERS.setdefault(path, function)


def declare_options_of_baseline_modules():
    """The options that the library's modules declare for the training regimes that switch the
    reference code's variants (``joint_init``, ``dele_code_form``, ...), on the baseline
    modules that the regimes are applied to in the pairs."""
    for name, attributes in (
            ("ELEmModule", dict(joint_init="ball", joint_init_roles="rel_embed",
                                dele_code_form="ball")),
            ("ELBEModule", dict(joint_init="box", joint_init_roles="rel_embed",
                                dele_code_form="box")),
            ("RegDELBEModule", dict(joint_init="box", joint_init_roles="rel_embed")),
            ("BoxSquaredELModule", dict(dele_code_form="box-with-bumps")),
            ("CanonicalMixtureModule", dict(not_a_region_model=(
                "classes are columns of a 0/1 membership matrix over the finite canonical "
                "domain (a set-based model, faithful by construction), not regions of R^d")))):
        for path in baseline_classes(name):
            cls = _import(path)
            for attribute, value in attributes.items():
                if attribute not in vars(cls):
                    setattr(cls, attribute, value)


def register_baseline_scorers():
    """The centre-distance scorers of the library's evaluation for the baseline modules."""
    from mowl.evaluation import el_scoring
    for library_path, scorer in list(el_scoring.CENTRE_DISTANCE_SCORERS.items()):
        if library_path.startswith("mowl.baselines."):
            continue
        for path in baseline_classes(library_path.rsplit(".", 1)[-1]):
            el_scoring.CENTRE_DISTANCE_SCORERS.setdefault(path, scorer)


def _import(path):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


@_adapter("ELEmModule")
def _elem(module):
    return methods.elem(
        None, None, embed_dim=module.embed_dim, margin=module.margin,
        reg_norm=module.reg_norm, all_negatives=module.all_negative_losses,
        linear_negative=module.negative_loss_form == "linear", epsilon=module.epsilon,
        gci1_radius_term=module.gci1_radius_term, tables=module)


@_adapter("ELBEModule")
def _elbe(module):
    return methods.elbe(None, None, margin=module.margin,
                        all_negatives=module.all_negative_losses, epsilon=module.epsilon,
                        gci3=module.gci3_form, tables=module)


@_adapter("EmELPPModule")
def _emelpp(module):
    return _lib("ballfam_parts").emelpp(
        None, None, embed_dim=module.embed_dim, margin=module.margin,
        linear_negative=module.negative_loss_form == "linear",
        gci1_radius_term=module.intersection_radius_term,
        radius=module.radius_form, role_direction=module.role_direction,
        gci0_shift=lambda: getattr(module, "subclass_relation", None),
        tables=module)


@_adapter("EmELVarModule")
def _emelvar(module):
    return _lib("ballfam_parts").emelvar(
        None, None, embed_dim=module.embed_dim, margin=module.margin,
        linear_negative=module.negative_loss_form == "linear",
        gci1_radius_term=module.intersection_radius_term,
        radius=module.radius_form, role_direction=module.role_direction,
        sigma_reg_weight=module.sigma_reg_weight,
        gci0_shift=lambda: getattr(module, "subclass_relation", None),
        tables=module)


@_adapter("RegDELBEModule")
def _regd(module):
    lib = _lib("ballfam_parts")
    return lib.regd(None, None, embed_dim=module.embed_dim, **lib._regd_options(module))


@_adapter("RegDELEmModule")
def _regd_elem(module):
    lib = _lib("ballfam_parts")
    return lib.regd_elem(None, None, embed_dim=module.embed_dim, reg_norm=module.reg_norm,
                         gci1_radius_term=module.gci1_radius_term, radius=module.radius,
                         consistent_radius=module.consistent_radius,
                         **lib._regd_options(module))


@_adapter("TransBoxModule")
def _transbox(module):
    return _lib("boxrel_methods").transbox(
        None, None, embed_dim=module.embed_dim, margin=module.margin,
        enhanced=module.use_enhancement, reg_factor=module.reg_factor,
        formulation=module.formulation,
        reference_bottom_losses=module.reference_bottom_losses,
        strict_negatives=module.strict_negatives, tables=module)


@_adapter("BoxELModule")
def _boxel(module):
    from mowl.representation.geometry import ComposedGeometry
    lib = _lib("boxrel_methods")
    return lib._boxel_geometry(ComposedGeometry, lib.boxel_objective(0.0), None, None, 0, None,
                               module.temperature, lib._role_options(module), module,
                               (lib.LocationPenalty(1.0),))


@_adapter("BoxSELModule")
def _boxsel(module):
    lib = _lib("boxrel_methods")
    return lib.boxsel(None, None, temperature=module.temperature, beta=module.beta,
                      location_weight=module.location_weight,
                      volume_weight=module.volume_weight, tables=module,
                      **dict(zip(("scale", "role_chain_composition"),
                                 lib._role_options(module))))


@_adapter("ELKEmModule")
def _elkem(module):
    geometry = _lib("boxrel_methods").elkem(
        None, None, margin=module.margin, abox_threshold=module.abox_threshold,
        variant=module.variant, tables=module)
    # The module uses margin 0 in evaluation mode; the geometry starts in the module's mode
    # (the two modes are switched separately afterwards).
    return geometry.train(module.training)


@_adapter("BoxSquaredELModule")
def _box2el(module):
    return _lib("multibox_methods").box2el(
        None, None, embed_dim=module.embed_dim, margin=module.gamma,
        delta=module.delta, reg_factor=module.reg_factor,
        all_negatives=module.all_negative_losses, epsilon=module.epsilon, tables=module)


@_adapter("MultiBoxELModule")
def _multiboxel(module):
    """The geometry of the module in its own objective (``module.objective``); with the
    structural objective the geometry has the module's definition table, targets and
    witness points as they are *now* (call again after ``set_structural_tables``)."""
    structural = module.objective == "structural"
    return _lib("multibox_methods").multiboxel(
        None, None, embed_dim=module.embed_dim,
        margin=module.margin if structural else module.gamma,
        n_components=module.n_components, delta=module.delta,
        reg_factor=module.reg_factor, component_reg=module.component_reg,
        component_aggregation=module.component_aggregation,
        diversity_reg=module.diversity_reg, reading=module.objective,
        max_components=module.max_components,
        existence_weight=module.existence_weight, size_reg=module.size_reg,
        bot_agg=module.bot_agg, witness_weight=module.witness_weight,
        size_reg_without_tables=getattr(module, "size_reg_without_tables", False),
        structural_definitions=getattr(module, "structural_definitions", True),
        tables=module)


@_adapter("MultiELBEModule")
def _multielbe(module):
    return _lib("multibox_methods").multielbe(
        None, None, embed_dim=module.embed_dim, margin=module.margin,
        n_components=module.n_components, component_reg=module.component_reg,
        component_aggregation=module.component_aggregation,
        diversity_reg=module.diversity_reg, reading=module.objective, tables=module)


@_adapter("VolumeMultiBoxELModule")
def _mcmultibox(module):
    return _lib("multibox_methods").mcmultibox(None, None, embed_dim=module.embed_dim,
                                               tables=module)


@_adapter("OctagonELModule")
def _octagon(module):
    own = module.disjoint_margin != module.margin
    return _lib("shapes_methods").octagon(
        None, None, margin=module.margin,
        disjoint_margin=module.disjoint_margin if own else None,
        neg_margin=module.neg_margin, min_width=module.min_width, tables=module)


@_adapter("PrismModule")
def _prism(module):
    return _lib("shapes_methods").prism(
        None, None, margin=module.margin,
        reading=getattr(module, "objective", "axiom"),
        existence_weight=getattr(module, "existence_weight", 1.0),
        size_reg=getattr(module, "size_reg", 1e-3),
        empty_box_reading=getattr(module, "empty_box_reading", "signed"),
        tables=module)


@_adapter("EIKEModule")
def _eike(module):
    return _lib("shapes_methods").eike(None, None, tables=module)


@_adapter("Concept2BoxModule")
def _concept2box(module):
    return _lib("shapes_methods").concept2box(None, None, margin=0.0, tables=module)


@_adapter("DFALCModule")
def _dfalc(module):
    return _lib("logic_fuzzy").dfalc(None, None, tables=module)


@_adapter("FALCONModule")
def _falcon(module):
    return _lib("logic_fuzzy").falcon(None, None, t_norm=module.t_norm,
                                      max_measure=module.max_measure, tables=module)


@_adapter("ALConeModule")
def _alcone(module):
    return _lib("logic_fuzzy").alcone(None, None, anneal=None, nondegeneracy=0.0,
                                      tables=module)


@_adapter("E2RModule")
def _e2r(module):
    return _lib("logic_subspace").E2RGeometry(module)


@_adapter("OrderEmbeddingsModule")
def _order_embeddings(module):
    return _lib("hierarchy_methods").order_embeddings(None, margin=module.margin,
                                                      tables=module)


@_adapter("CatEModule")
def _cate(module):
    return _lib("hierarchy_methods").cate(None, tables=module)


@_adapter("BoxLatticeModule")
def _box_lattice(module):
    return _lib("hierarchy_methods").box_lattice(None, tables=module)


@_adapter("GumbelBoxModule")
def _gumbel_box(module):
    return _lib("hierarchy_methods").gumbel_box(None, tables=module)


@_adapter("BinaryCodeBoxModule")
def _bcbox(module):
    return _lib("hierarchy_methods").bcbox(None, variant=module.variant, tables=module)


@_adapter("HyperbolicConesModule")
def _hyperbolic_cones(module):
    return _lib("hierarchy_methods").hyperbolic_cones(None, tables=_WithLibraryManifold(module))


@_adapter("ShadowConesModule")
def _shadow_cones(module):
    return _lib("hierarchy_methods").shadow_cones(None, tables=_WithLibraryManifold(module))


@_adapter("DiskEmbeddingsModule")
def _disk_embeddings(module):
    return _lib("hierarchy_methods").disk_embeddings(None, tables=_WithLibraryManifold(module))


@_adapter("HiTModule")
def _hit(module):
    return _lib("hierarchy_methods").hit(tables=module)


@_adapter("OnTModule")
def _ont(module):
    return _lib("hierarchy_methods").ont(tables=module)


def _query(builder):
    def adapter(module):
        lib = _lib("query_methods")
        return getattr(lib, builder)(tables=module, num_classes=lib._number_of_classes(module))
    return adapter


for _name, _builder in (("Query2BoxGeometry", "query2box"), ("BetaEGeometry", "betae"),
                        ("ConEGeometry", "cone_qe"), ("AConEGeometry", "acone")):
    _adapter(_name)(_query(_builder))


# geometries of the library that read a module given to them
_adapter("GroundedELModule")(_lib("other_regions").GroundedGeometry)
_adapter("CanonicalMixtureModule")(_lib("other_regions").CanonicalMixtureGeometry)
_adapter("RegionKGEModel")(_lib("other_graph").PyKEENGeometry)


mirror_library_adapters()
register_baseline_scorers()

_DECLARED = []


def ensure_baseline_options():
    """:func:`declare_options_of_baseline_modules`, once, as soon as the JVM is up (the
    baseline modules cannot be imported before)."""
    import jpype
    if not _DECLARED and jpype.isJVMStarted():
        _DECLARED.append(True)
        declare_options_of_baseline_modules()
