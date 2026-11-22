"""The owner's rule, audited: everything a user can pick as a PART (class model, relation model,
measure, objective option) defaults to the correct reading; a published / old / "as released"
reading of a part is an option that the builder of the named method selects.

``PUBLISHED`` is the table of every constructor option of a registered part whose docstring or
value names such a reading. One row = one published value:

    (kind, part, option, default, published value, builder, builder options, where to read it)

and the tests assert, row by row, that (1) the default in the signature is the one in the table,
(2) it is NOT the published value, (3) the builder of the named method, called with its options,
gives the part the published value (so that the published reading is reachable and selected where
the method needs it), and (4) the option is a real option of the part. ``NEUTRAL`` lists every
other option (sizes, tables, initialisation, hyper-parameters that are not a reading of an
axiom). The completeness test fails when a new option appears in a signature of a registered
class model, relation model, measure or of ``Objective`` that is in neither: decide, then add
it to the table (a published / old reading) or to ``NEUTRAL``.

The table is written as Markdown to ``notes/part-defaults.md`` by the maintainers of the audit
(``python -m tests.representation.test_defaults_are_correct`` prints it).
"""
import inspect
from unittest import TestCase

import mowl
mowl.init_jvm("1g")

import mowl.representation as R
from mowl.representation.objectives import Objective

# (kind, part, option, default, published, builder, builder options, path of the value)
# kind: "class", "relation", "measure" (registry name of the part) or "objective" (a class).
# path: attribute path from the geometry of the builder; "classes.x" = geometry.class_model.x.
CLASS, RELATION, MEASURE, OBJECTIVE = "class", "relation", "measure", "objective"

PUBLISHED = [
    # ---- class models -------------------------------------------------------------------
    (CLASS, "ball", "radius", "abs", "relu", "emelpp", {"radius": "relu"},
     "classes.radius_form"),
    (CLASS, "ball", "radius", "abs", "softplus", "regd_elem", {"radius": "softplus"},
     "classes.radius_form"),
    (CLASS, "half-axis cone", "constant_logit", 1.0, 1e4, "alcone", {"constant_logit": 1e4},
     "classes.constant_logit"),
    # ---- relation models ----------------------------------------------------------------
    (RELATION, "affine map", "scale", "positive", "signed", "boxel", {"scale": "signed"},
     "relations.scale"),
    (RELATION, "affine map", "role_chain_composition", "in_order", "as_released", "boxel",
     {"role_chain_composition": "as_released"}, "relations.role_chain_composition"),
    (RELATION, "affine map", "role_scale", "equal", "at_most", "boxel", {},
     "relations.role_scale"),
    (RELATION, "affine map with range and error box", "sound", True, False, "elkem",
     {"variant": "reference"}, "relations.sound"),
    (RELATION, "angle rotation", "aperture", "clip", "wrap", "acone", {},
     "relations.aperture"),
    (RELATION, "angle rotation", "aperture_change", "tolerance", "signed", "acone", {},
     "relations.aperture_change"),
    (RELATION, "box of translations", "activation", "abs", "relu", "query2box", {},
     "relations.activation_name"),
    (RELATION, "box of translations", "activation", "abs", "none", "query2box",
     {"activation": "none"}, "relations.activation_name"),
    (RELATION, "box of translations", "clamp_extent", True, False, "query2box", {},
     "relations.clamp_extent"),
    (RELATION, "box of translations", "convention", "y-x", "x-y", "transbox", {},
     "relations.convention"),
    (RELATION, "box of translations", "init_range", None, "number", "query2box", {},
     "relations.init_range_is_set"),
    (RELATION, "translation", "direction", None, "row", "emelpp", {}, "relations.direction"),
    (RELATION, "translation", "direction", None, "batch", "emelpp",
     {"role_direction": "batch"}, "relations.direction"),
    (RELATION, "translation with tolerance", "direction", None, "row", "emelvar", {},
     "relations.direction"),
    # ---- measures -----------------------------------------------------------------------
    (MEASURE, "bounds per coordinate", "separation", "any", "every", "elbe", {},
     "measure.separation"),
    (MEASURE, "bounds per coordinate", "extent", "thinnest", "norm", "elbe", {},
     "measure.extent"),
    (MEASURE, "bounds per coordinate", "conjunction", "exact", "nonempty", "elbe", {},
     "measure.conjunction"),
    (MEASURE, "bounds per coordinate", "empty_boxes", "empty", "formula", "regd", {},
     "measure.empty_boxes"),
    (MEASURE, "averaged bounds", "extent", "thinnest", "norm", "elkem", {},
     "measure.extent"),
    (MEASURE, "averaged bounds", "conjunction", "exact", "nonempty", "elkem", {},
     "measure.conjunction"),
    (MEASURE, "averaged bounds", "empty_boxes", "empty", "formula", "elkem", {},
     "measure.empty_boxes"),
    (MEASURE, "masked bounds", "separation", "any", "every", "transbox", {},
     "measure.separation"),
    (MEASURE, "masked bounds", "empty_weight", 1.0, 0.5, "transbox", {},
     "measure.empty_weight"),
    (MEASURE, "masked bounds", "gci1", "inclusion", "overlap", "transbox",
     {"formulation": "reference"}, "measure.gci1"),
    (MEASURE, "boundary and depth", "depth_on_containment", False, True, "regd", {},
     "measure.depth_on_containment"),
    (MEASURE, "centre distance", "conjunction", "bounding", "published", "elem", {},
     "measure.conjunction"),
    (MEASURE, "centre distance", "radius_term", True, False, "elem",
     {"gci1_radius_term": False}, "measure.radius_term"),
    (MEASURE, "conflict", "stabiliser", "clamp", "add", "falcon", {}, "measure.stabiliser"),
    (MEASURE, "cosine order", "norm_term", "hinge", "signed", "eike", {},
     "measure.norm_term"),
    (MEASURE, "ellipsoid", "containment", "bounding", "published", "eike", {},
     "measure.containment"),
    (MEASURE, "log volume ratio", "zero_at_inclusion", True, False, "box_lattice",
     {"zero_at_inclusion": False}, "measure.zero_at_inclusion"),
    (MEASURE, "log volume ratio", "tail", "exact", "linear", "gumbel_box", {},
     "measure.tail"),
    (MEASURE, "log volume ratio", "tail", "exact", "floor", "gumbel_box",
     {"log_guard": "epsilon"}, "measure.tail"),
    (MEASURE, "log volume ratio", "energy_floor", 1e-12, 1e-9, "gumbel_box",
     {"log_guard": "epsilon"}, "measure.energy_floor"),
    (MEASURE, "point-to-region distance", "alpha", 0.0, 0.5, "concept2box", {},
     "measure.alpha_of_cross_view"),
    (MEASURE, "point-to-region distance", "squash", False, True, "concept2box", {},
     "measure.squash_of_cross_view"),
    (MEASURE, "sampled volume", "aggregate", "min", "mean", "mcmultibox",
     {"aggregate": "mean"}, "measure.aggregate"),
    (MEASURE, "shadow cone", "signed", False, True, "shadow_cones", {}, "measure.signed"),
    (MEASURE, "shadow cone", "negative_form", "hinge", "energy", "shadow_cones",
     {"negative_form": "energy"}, "measure.negative_form"),
    (MEASURE, "protrusion", "signed", False, True, "disk_embeddings", {}, "measure.signed"),
    (MEASURE, "angle", "apex", "inside", "formula", "hyperbolic_cones", {},
     "measure.apex"),
    (MEASURE, "union of bounds", "meets_reading", "some", "every", "multiboxel", {},
     "measure.meets_reading"),
    (MEASURE, "union of bounds", "meets_reading", "some", "every", "multielbe", {},
     "measure.meets_reading"),
    (MEASURE, "volume share", "strict_margin", "bounds", "ignore", "boxel", {},
     "measure.strict_margin"),
    (MEASURE, "volume share", "strict_margin", "bounds", "ignore", "boxsel", {},
     "measure.strict_margin"),
    (MEASURE, "prism bounds", "empty", "any", "triangle", "prism", {},
     "measure.empty_form"),
    (MEASURE, "prism bounds", "conjunction", "exact", "nonempty", "prism", {},
     "measure.conjunction"),
    (MEASURE, "prism bounds", "empty_prisms", "empty", "formula", "prism", {},
     "measure.empty_prisms"),
    (MEASURE, "bounds per coordinate over slots", "reading", "exact", "structural",
     "multiboxel", {"reading": "structural"}, "measure.reading"),
    # ---- the objective ------------------------------------------------------------------
    (OBJECTIVE, "Objective", "gci3", "inclusion", "overlap", "elbe", {},
     "objective.gci3_reading"),
    (OBJECTIVE, "Objective", "gci2_negative", "apart", "overlap", "elbe", {},
     "objective.gci2_negative"),
    (OBJECTIVE, "Objective", "gci2_negative", "apart", "delta", "box2el", {},
     "objective.gci2_negative"),
    (OBJECTIVE, "Objective", "gci2_negative", "apart", "shrunk", "prism",
     {"gci2_negative_triangle": "inradius"}, "objective.gci2_negative"),
    (OBJECTIVE, "Objective", "negatives", "all", "gci2", "elbe", {}, "objective.negatives"),
    (OBJECTIVE, "Objective", "strict_negatives", True, False, "elbe", {},
     "objective.strict_negatives"),
    (OBJECTIVE, "Objective", "slack", False, True, "elem", {}, "objective.slack"),
    (OBJECTIVE, "Objective", "empty", "size", "raw", "elem", {}, "objective.empty"),
    (OBJECTIVE, "Objective", "gci2_offset", 0.0, 1e-5, "elem", {},
     "objective.gci2_offset"),
    (OBJECTIVE, "Objective", "negative_margin", "same", "strict", "elem", {},
     "objective.negative_margin"),
    (OBJECTIVE, "Objective", "gci2_reading", "exists", "image", "boxel", {},
     "objective.gci2_reading"),
    (OBJECTIVE, "Objective", "bottom", "empty", "class", "boxel", {}, "objective.bottom"),
    (OBJECTIVE, "Objective", "gci3_bot_negative", None, "filler", "box2el", {},
     "objective.gci3_bot_negative"),
    (OBJECTIVE, "Objective", "gci1_bot", None, "product", "boxel", {},
     "objective.gci1_bot_reading"),
    (OBJECTIVE, "Objective", "squared", False, "forms", "box2el", {},
     "objective.squared_is_set"),
    (OBJECTIVE, "Objective", "unit_norm", False, "forms", "emelpp", {},
     "objective.unit_norm_is_set"),
    (OBJECTIVE, "Objective", "freeze_size_negatives", False, True, "box_lattice", {},
     "objective.freeze_size_negatives"),
    # objectives of named methods that carry an option of a published reading
    (OBJECTIVE, "ELKEmObjective", "trivial_disjointness", True, False, "elkem",
     {"variant": "reference"}, "objective.trivial_disjointness"),
    (OBJECTIVE, "StructuralReading", "empty_regions", "keep", "abs", "prism",
     {"reading": "structural", "empty_box_reading": "absolute"}, "objective.empty_regions"),
    (OBJECTIVE, "StructuralGeometry", "size_reg_without_tables", False, True, "multiboxel",
     {"reading": "structural", "size_reg_without_tables": True}, "size_reg_without_tables"),
    (OBJECTIVE, "StructuralGeometry", "size_reg_without_tables", False, True, "prism",
     {"reading": "structural", "size_reg_without_tables": True}, "size_reg_without_tables"),
]

# Options that are neutral: sizes and tables, initialisation, hyper-parameters, labels and
# options that switch a feature on (default: the feature is off). A published reading is
# never in this list. Per part ("kind:name" or the class name), plus GENERIC for all.
GENERIC = {"size", "embed_dim", "dim", "centres", "offsets", "radii", "table", "tables", "seed",
           "bumps", "vectors", "base", "host", "init_std", "init_scale", "init_max_norm",
           "init_radius", "init", "scale_init", "eps", "epsilon", "margin", "bound", "generator"}
NEUTRAL = {
    "class:ball": set(),
    "class:box with fixed classes": {"bot_mask", "top_mask", "big"},
    "class:box with lower corner": {"mins", "deltas", "min_bounds", "delta_bounds"},
    "class:disk": {"manifold"},
    "class:ellipsoid": {"axes_raw", "radius_raw"},
    "class:fuzzy set": set(), "class:fuzzy set (lukasiewicz)": set(),
    "class:fuzzy set (product)": set(),
    "class:geodesic cone": {"K", "manifold"},   # K = 0.1 is the paper's hyper-parameter
    "class:half-axis cone": {"logits", "family", "top_index", "bottom_index"},
    "class:manifold point": {"manifold"},
    "class:octagon product": set(),
    "class:order cone": {"nonnegative"},        # a convention: order embeddings vs CatE
    "class:point": {"family", "scale", "bump"},
    "class:sector cone": {"apertures"},
    "class:soft box": {"beta", "offset"},       # offset: two ways to keep a width positive
    "class:triangle x box": {"tri_centres", "tri_inradii", "box_centres", "box_offsets",
                             "resolver"},
    "class:union of boxes": {"components", "bump"},
    "class:union of boxes with slots": {"components", "max_components", "bump"},
    "class:unit box": {"min_delta", "init_side", "soft"},
    "class:box with bump": set(), "class:box": set(),
    "relation:affine map": {"scales", "eps"},
    "relation:affine map with range and error box": {"range_centres", "range_offsets",
                                                     "error_offsets", "slopes"},
    "relation:angle rotation": {"angle_range", "inverse_of"},
    "relation:box of translations": {"offsets", "inverse_of"},
    "relation:fuzzy relation": {"domain"},
    "relation:head and tail boxes": {"shape", "head_centres", "head_offsets", "tail_centres",
                                     "tail_offsets"},
    # the interaction functions of PyKEEN's BoxE / ExpressivE, with PyKEEN's defaults
    "relation:head and tail boxes (distance score)": {"tanh_map", "p", "power_norm",
                                                      "subclass_id"},
    "relation:incidence matrix": set(),
    "relation:region over pairs": {"p", "tanh_map", "min_denom", "subclass_id"},
    "relation:translation": set(),
    "relation:translation with tolerance": {"tolerances", "sigma_init"},
    "relation:union of boxes over pairs": {"components"},
    "measure:angle": {"signed", "negative_form"},
    "measure:averaged bounds": {"separation"},
    "measure:boundary and depth": {"lamda", "p", "gamma1", "gamma2"},
    "measure:bounds per coordinate over slots": set(),
    "measure:bounds per coordinate": set(),
    "measure:centre distance": set(),
    "measure:conflict": {"max_measure"},
    "measure:cosine order": set(),
    "measure:distance and depth": {"centri_weight"},
    "measure:divergence": set(),
    "measure:ellipsoid": set(),
    "measure:gumbel volume ratio": {"beta", "power"},
    "measure:half-axis hinge": {"empty_margin", "tau", "source"},
    "measure:log volume ratio": {"intersection_temp", "volume_temp", "disjoint_weight",
                                 "gumbel_shift", "dtype", "separation"},
    "measure:masked bounds": {"separation"},
    "measure:mean violation": set(),
    "measure:octagon bounds": set(),
    "measure:order violation": {"direction", "p", "negative_form"},
    "measure:point distance": {"power"},
    "measure:point-to-region distance": {"beta", "weighted"},
    "measure:prism bounds": set(),
    "measure:protrusion": {"negative_form"},
    "measure:sampled volume": {"points", "num_points", "temperature", "chunk_size"},
    "measure:sector angle": set(),
    "measure:shadow cone": {"formulation", "radius", "horosphere_height", "gamma3"},
    "measure:soft membership": {"beta"},
    "measure:triple score": {"relations"},
    "measure:union of bounds": {"aggregation", "tau"},
    "measure:volume share": {"temperature", "floor", "ceiling"},
    "Objective": {"epsilon", "disjoint_margin", "empty_margin", "neg_margins", "delta",
                  "weights", "ranking", "tolerance_weight", "gci0_shift", "inside_margin",
                  "gci0_negative", "gci3_bot", "slack_unused"},
}
# measure options kept neutral with a reason, to be read by a human:
#   zero ``centri_weight``, ``lamda``, ``gamma1/2``: the weights of a paper's loss (0 or 1 is
#   the plain reading, the paper's values are chosen by the method builders);
#   ``gci0_negative``: "apart" / "negative" / "outside": three negative readings of C below D,
#   each right for its family (hierarchy methods and RegD choose theirs);
#   ``gci3_bot``: "exists" / "filler": two equivalent readings of exists r.C below bottom.


# Options whose docstring names a published thing and that are neutral all the same, with the
# reason (the completeness test below reads the docstrings for such words).
NEUTRAL_THOUGH_NAMED = {
    ("measure", "log volume ratio", "dtype"): "precision of the computation, not a reading",
    ("class", "geodesic cone", "K"): "hyper-parameter, 0.1 in the paper",
    ("measure", "boundary and depth", "gamma1"): "weight of the paper's loss",
    ("measure", "boundary and depth", "gamma2"): "weight of the paper's loss",
    ("measure", "log volume ratio", "gumbel_shift"): "two published temperatures (Dasgupta et "
                                                     "al., Zhang et al.), both sound",
    ("measure", "volume share", "floor"): "numerical clip of the volumes, as in BoxEL's code",
    ("measure", "volume share", "ceiling"): "numerical clip of the volumes, as in BoxEL's code",
}


def _signature_options(factory):
    target = factory if inspect.isfunction(factory) else factory.__init__
    out = {}
    for p in inspect.signature(target).parameters.values():
        if p.name == "self" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        out[p.name] = None if p.default is inspect._empty else p.default
    return out


def _registered():
    for kind, registry in ((CLASS, R.CLASS_MODELS), (RELATION, R.RELATION_MODELS),
                           (MEASURE, R.MEASURES)):
        for name, factory in sorted(registry.items()):
            yield kind, name, factory
    yield OBJECTIVE, "Objective", Objective


def _neutral_key(kind, name):
    return name if kind == OBJECTIVE else f"{kind}:{name}"


# -- reading the value a builder gave a part -------------------------------------------------

def _geometry(method, options):
    builder = R.METHODS[method]
    arguments = dict(embed_dim=6)
    if method in ("acone", "cone_qe", "query2box", "betae"):
        arguments.update(nb_classes=6, nb_relations=2)
    else:
        arguments.update(nb_classes=10, nb_relations=3)
    arguments.update({"falcon": {"anon_domain": 3}}.get(method, {}))
    arguments.update(options)
    nb_classes, nb_relations = arguments.pop("nb_classes"), arguments.pop("nb_relations")
    return builder(nb_classes, nb_relations, **arguments)


def _read(geometry, path, part=None, kind=None):
    head, _, rest = path.partition(".")
    if head == "size_reg_without_tables":
        return geometry.size_reg_without_tables
    owner = {"classes": lambda g: g.class_model,
             "relations": lambda g: getattr(g, "relation_model", None) or g.relations,
             "measure": lambda g: g.measure,
             "objective": lambda g: g.objective}[head](geometry)
    if kind in (CLASS, RELATION, MEASURE) and not hasattr(owner, rest.split(".")[0]):
        registry = {CLASS: R.CLASS_MODELS, RELATION: R.RELATION_MODELS,
                    MEASURE: R.MEASURES}[kind]
        found = _find(owner, registry[part], 0) if inspect.isclass(registry[part]) else None
        owner = found if found is not None else owner
    special = {
        "activation_name": lambda o: next(n for n, f in _ACTIVATIONS().items()
                                          if o.activation is f),
        "init_range_is_set": lambda o: "number" if _is_uniform(o) else None,
        "alpha_of_cross_view": lambda o: _cross_view(geometry).alpha,
        "squash_of_cross_view": lambda o: _cross_view(geometry).squash,
        "squared_is_set": lambda o: "forms" if o.squared else False,
        "unit_norm_is_set": lambda o: "forms" if o.unit_norm else False,
        "convention": lambda o: "y-x" if o.sign > 0 else "x-y",
        "radius_term": lambda o: o.radius_term,
        "gci3_reading": lambda o: o.gci3_reading,
    }
    if rest in special:
        return special[rest](owner)
    return getattr(owner, rest)


def _ACTIVATIONS():
    from mowl.representation.library.boxrel_relations import _ACTIVATIONS as table
    return table


def _is_uniform(relations):
    # Query2Box initialises the translations uniformly in [-r, r] (a number), others at norm 1
    import torch as th
    v = relations.vectors.weight.detach()
    return not bool(th.allclose(v.norm(dim=1), th.ones(len(v)), atol=1e-4))


def _cross_view(geometry):
    PointToRegion = R.MEASURES["point-to-region distance"]
    for owner in (geometry.measure, geometry):
        for value in vars(owner).values():
            found = _find(value, PointToRegion, 0)
            if found is not None:
                return found
    raise AssertionError("no point-to-region measure in the geometry")


def _find(value, cls, depth):
    if isinstance(value, cls):
        return value
    if depth > 3:
        return None
    items = value.values() if isinstance(value, dict) else \
        value if isinstance(value, (list, tuple)) else \
        vars(value).values() if hasattr(value, "__dict__") else ()
    for item in items:
        found = _find(item, cls, depth + 1)
        if found is not None:
            return found
    return None


def unlisted_options(options, listed, neutral_table):
    """The options of ``options[(kind, name)]`` in neither ``listed`` (the table) nor the
    neutral entries, as messages."""
    problems = []
    for (kind, name), signature in sorted(options.items()):
        neutral = neutral_table.get(_neutral_key(kind, name))
        if neutral is None:
            problems.append(f"{kind} {name}: not in NEUTRAL")
            continue
        for option in signature:
            if option in GENERIC or option in neutral or option in listed.get((kind, name), ()):
                continue
            problems.append(f"{kind} {name}: option {option!r} (default {signature[option]!r}) "
                            f"is in neither PUBLISHED nor NEUTRAL")
    return problems


class TestDefaultsAreCorrect(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.options = {(kind, name): _signature_options(factory)
                       for kind, name, factory in _registered()}
        from mowl.representation.library.boxrel_objectives import ELKEmObjective
        from mowl.representation.library.structural_methods import StructuralGeometry
        from mowl.representation.library.structural_reading import StructuralReading
        extra = {"ELKEmObjective": ELKEmObjective, "StructuralReading": StructuralReading,
                 "StructuralGeometry": StructuralGeometry}
        for name, factory in extra.items():
            cls.options[(OBJECTIVE, name)] = _signature_options(factory)

    def test_the_table_names_real_options_with_their_real_defaults(self):
        for kind, part, option, default, published, method, _, _ in PUBLISHED:
            options = self.options.get((kind, part))
            self.assertIsNotNone(options, f"{kind} {part} is not registered")
            self.assertIn(option, options, f"{part} has no option {option!r}")
            self.assertEqual(options[option], default,
                             f"the default of {part}.{option} is not the one in the table")
            self.assertIn(method, R.METHODS, method)

    def test_no_default_is_the_published_reading(self):
        for kind, part, option, default, published, method, _, _ in PUBLISHED:
            self.assertNotEqual(default, published,
                                f"{part}.{option} defaults to the published value {published!r}")
            self.assertFalse(default is published and default is not None,
                             f"{part}.{option}")

    def test_the_named_methods_select_the_published_reading(self):
        for kind, part, option, default, published, method, kwargs, path in PUBLISHED:
            geometry = _geometry(method, kwargs)
            got = _read(geometry, path, part, kind)
            self.assertEqual(got, published,
                             f"{method}({kwargs}) should give {part}.{option} = "
                             f"{published!r}, gives {got!r}")

    def test_every_option_of_every_part_is_in_the_table_or_neutral(self):
        listed = {}
        for kind, part, option, *_ in PUBLISHED:
            listed.setdefault((kind, part), set()).add(option)
        parts = {key: value for key, value in self.options.items()
                 if key[1] not in ("ELKEmObjective", "StructuralReading", "StructuralGeometry")}
        self.assertEqual(unlisted_options(parts, listed, NEUTRAL), [])

    def test_a_new_option_is_caught(self):
        # the check itself: an option that is in neither the table nor the neutral list is
        # reported, and so is a part that has no neutral entry at all
        class Fake:
            def __init__(self, size=None, shiny_new_option="old reading"):
                pass
        options = {(MEASURE, "fake"): _signature_options(Fake)}
        self.assertEqual(len(unlisted_options(options, {}, {"measure:fake": set()})), 1)
        self.assertEqual(len(unlisted_options(options, {}, {})), 1)
        self.assertEqual(unlisted_options(options, {(MEASURE, "fake"): {"shiny_new_option"}},
                                          {"measure:fake": set()}), [])
        self.assertEqual(unlisted_options(options, {}, {"measure:fake": {"shiny_new_option"}}),
                         [])

    def test_the_neutral_lists_name_real_options(self):
        # a stale entry (an option that no longer exists) would hide a new one
        stale = []
        for (kind, name), options in self.options.items():
            neutral = NEUTRAL.get(_neutral_key(kind, name), set())
            stale += [f"{name}.{o}" for o in neutral if o not in options and o != "slack_unused"]
        self.assertEqual(stale, [])

    def test_the_table_is_complete_for_the_options_the_docstrings_name(self):
        # every option whose docstring names a published / old / "as released" reading is
        # in the table (a neutral option may not mention one)
        import re
        words = re.compile(r"as_released|as released|published|earlier|authors' code|"
                           r"reference code|original modules|the old|released code",
                           re.IGNORECASE)
        in_table = {(kind, part, option) for kind, part, option, *_ in PUBLISHED}
        missing = []
        for (kind, name), options in sorted(self.options.items()):
            if name in ("ELKEmObjective", "StructuralReading", "StructuralGeometry"):
                continue
            factory = dict(((k, n), f) for k, n, f in _registered())[(kind, name)]
            doc = (inspect.getdoc(factory) or "") + "\n" + \
                (inspect.getdoc(factory.__init__) if inspect.isclass(factory) else "")
            for option in options:
                m = re.search(r":param\s+(?:[^:]*,\s*)?" + re.escape(option)
                              + r"\b[^:]*:(.*?)(?=\n\s*:param|\n\n|\Z)", doc, re.S)
                if m and words.search(m.group(1)) and (kind, name, option) not in in_table \
                        and (kind, name, option) not in NEUTRAL_THOUGH_NAMED:
                    missing.append(f"{kind} {name}.{option}: {' '.join(m.group(1).split())[:90]}")
        self.assertEqual(missing, [])


def markdown():
    lines = ["# Part defaults and published readings", "",
             "Every option of a registered class model, relation model, measure and of "
             "`Objective` whose docstring or value names a published / old / \"as released\" "
             "reading. The default is the correct reading; the published one is an option "
             "that the builder of the named method selects (a method called with its options "
             "in the last column).", "",
             "| Part | Option | Default | Published value | Selected by |",
             "|---|---|---|---|---|"]
    for kind, part, option, default, published, method, kwargs, _ in PUBLISHED:
        by = f"`{method}`" + (f" with `{kwargs}`" if kwargs else " (by default)")
        lines.append(f"| {kind}: {part} | `{option}` | `{default!r}` | `{published!r}` | {by} |")
    lines += ["", "Notes:", "",
              "* `boundary and depth.depth_on_containment`: the part's default `False` is both "
              "the correct reading (a contained region costs nothing) and the authors' released "
              "code; the paper's formula is `True`, which `regd` and `regd_elem` select.",
              "* `masked bounds.empty_weight`: 1.0 is the authors' code; 0.5 is what mOWL's "
              "TransBox module had.",
              "* `Objective.squared` and `unit_norm`: the published value is a tuple of forms "
              "(Box2EL squares six forms; EmEL++ adds the norm term to nine), shown as `forms`.",
              "* Options judged neutral (sizes, tables, initialisation, hyper-parameters, "
              "conventions that are both right, features that are off by default) are the "
              "`NEUTRAL` table of `tests/representation/test_defaults_are_correct.py`; the "
              "test fails for an option that is in neither table."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(markdown())
