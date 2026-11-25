"""Octagon, prism, EIKE and Concept2Box: the combination of parts users get
(``mowl.models``) against the direct implementation (``mowl.baselines``)."""
import importlib

import torch as th

from tests.baselines.harness import Pair

EMBED = 8

#: The tables of each method: attributes of the module, with the same names on the direct
#: module and on the combination.
TABLES = {
    "octagon": ("class_embed", "class_radius", "rel_embed", "ind_embed"),
    "prism": ("tri_center", "tri_inradius", "box_center", "box_offset", "rel_embed",
              "ind_embed"),
    "eike": ("class_center", "class_axes_raw", "class_radius_raw", "class_intensional",
             "ind_embed", "rel_embed", "M"),
    "concept2box": ("class_embed", "ontology_rel_embed", "rel_embed", "ind_embed"),
}

#: Where the class of each method lives, in the library and in the baselines.
CLASSES = {
    "octagon": ("OctagonEL", "mowl.baselines.models.octagon.model"),
    "prism": ("PrismEL", "mowl.baselines.models.prism.model"),
    "eike": ("EIKE", "mowl.baselines.models.eike.model"),
    "concept2box": ("Concept2Box", "mowl.baselines.models.concept2box.model"),
}


def _weight(module, name):
    value = getattr(module, name)
    return value.weight if hasattr(value, "weight") else value


def _present(old, new, name):
    """Whether the table exists (no individuals, no table); both sides must agree."""
    a, b = getattr(old.module, name), getattr(new.module, name)
    assert (a is None) == (b is None), name
    return a is not None


def _network(module):
    """The two-layer network of Concept2Box (``project`` in the direct module)."""
    return module.box_network if hasattr(module, "box_network") else module.project


def _copy(names, network):
    def copy(old, new):
        with th.no_grad():
            for name in names:
                if _present(old, new, name):
                    _weight(new.module, name).copy_(_weight(old.module, name))
            if network:
                _network(new.module).load_state_dict(_network(old.module).state_dict())
    return copy


def _parameters(names, network):
    def parameters(model):
        out = {name: _weight(model.module, name).detach().clone() for name in names
               if getattr(model.module, name) is not None}
        if network:
            for key, value in _network(model.module).state_dict().items():
                value = value.detach().clone()
                if key == "2.bias":
                    # The first half of this bias (the centre) has a gradient that is zero
                    # in theory; its drift is rounding noise times Adam's step and is
                    # checked on its own (test_concept2box_bias_noise.py). The offset half
                    # is compared like every other parameter.
                    value = value[value.numel() // 2:]
                out[f"network.{key}"] = value
        return out
    return parameters


def _scores(model):
    """What a ranking evaluation reads: the module's cost of every subsumption between two
    classes, of every existential axiom with the first relation (where the method reads
    it), and of the assertions."""
    module = model.module
    n = len(model.class_index_dict)
    a = th.arange(n).repeat_interleave(n)
    b = th.arange(n).repeat(n)
    out = {"gci0": module(th.stack([a, b], dim=1), "gci0")}
    if "gci2" in model.training_datasets or "gci2" in getattr(model, "scored_forms", ()):
        out["gci2"] = module(th.stack([a, th.zeros_like(a), b], dim=1), "gci2")
    m = len(model.individual_index_dict)
    if m:
        i = th.arange(m).repeat_interleave(n)
        c = th.arange(n).repeat(m)
        out["class_assertion"] = module(th.stack([i, c], dim=1), "class_assertion")
        x = th.arange(m).repeat_interleave(m)
        y = th.arange(m).repeat(m)
        out["object_property_assertion"] = module(
            th.stack([x, th.zeros_like(x), y], dim=1), "object_property_assertion")
    # every other form of the training data, on its first rows
    for form, data in model.training_datasets.items():
        if form not in out and len(data):
            out[form] = module(data[:12], form)
    return out


def _with_structural_tables(model):
    """The direct implementation builds the structural tables when it is trained; the
    combination builds them at once. Build them now for the direct one, so that the losses
    are compared before training (this has nothing to do with the order of rows)."""
    model._load_datasets()
    if getattr(model, "objective", None) == "structural":
        model._structural_tables = None
        if hasattr(model, "_attach_structural_tables"):
            model._attach_structural_tables()
        else:
            from mowl.baselines.nn.el.structural import build_structural_tables
            model._structural_tables = build_structural_tables(
                model, k=1, max_components=2 ** 30, enabled=model.structural_definitions,
                max_level=model.max_depth, cm=model.canonical_model)
            table, targets, top_idx = model._structural_tables
            model.module.set_structural_tables(table, targets, top_idx=top_idx)
    return model


def _abox():
    from tests.models.kg_region_data import abox_dataset
    return abox_dataset()


#: Defaults of the combination that differ from the direct module on purpose: the pair is
#: built with the values of the direct module (Concept2Box: the Gumbel scale 1 and the
#: initial scale 0.5, with which the model does not learn beyond a few tens of dimensions).
DIRECT_VALUES = {"concept2box": {"beta": 1.0, "init_scale": 0.5}}


def _pair(method, variant=None, dataset=None, **options):
    name, baseline = CLASSES[method]

    def new(dataset):
        return getattr(importlib.import_module("mowl.models"), name)(
            dataset, embed_dim=EMBED, **{**DIRECT_VALUES.get(method, {}), **options})

    def old(dataset):
        return _with_structural_tables(getattr(importlib.import_module(baseline), name)(
            dataset, embed_dim=EMBED, **options))

    network = method == "concept2box"
    atol = 1e-5
    return Pair(key=method if variant is None else f"{method}/{variant}", new=new, old=old,
                copy=_copy(TABLES[method], network), scores=_scores,
                parameters=_parameters(TABLES[method], network), atol=atol,
                **({} if dataset is None else {"dataset": dataset}))


# Parameters that do not move in a TBox-only pair, and why (checked by test_parameters_move.py):
# EIKE and Concept2Box use their ``ind_embed`` and ``rel_embed`` for assertions only
# (``object_property_assertion_loss``; TransE-like), EIKE's ``M`` only with
# ``instance_map="matrix"`` in class assertions. The pairs with ``_abox`` data exercise them:
# eike/abox (ind_embed, rel_embed), eike/matrix-abox (M), concept2box/abox (rel_embed).
PAIRS = [
    _pair("octagon", learning_rate=0.01),
    _pair("octagon", "own-margins-and-decay", learning_rate=0.01, disjoint_margin=0.3,
          neg_margin=0.2, min_width=0.05, lr_decay=0.9),
    _pair("prism", learning_rate=0.01),
    _pair("prism", "structural", learning_rate=0.01, objective="structural"),
    # EIKE ranks classes and assertions, so these two read data with individuals and roles:
    # the pair compares the subsumption score, the class assertion score and the role
    # assertion score (on the family ontology alone, only the first exists).
    _pair("eike", dataset=_abox, learning_rate=0.01),
    _pair("eike", "matrix", _abox, learning_rate=0.01, instance_map="matrix", alpha=0.5,
          margins=(0.5, 0.3, 1.0)),
    _pair("concept2box", learning_rate=0.01),
    _pair("concept2box", "triple", learning_rate=0.01, gci2_reading="triple"),
    # with individuals: class assertions and role assertions
    _pair("octagon", "abox", _abox, learning_rate=0.01),
    _pair("prism", "abox", _abox, learning_rate=0.01),
    _pair("prism", "structural-abox", _abox, learning_rate=0.01, objective="structural"),
    _pair("eike", "abox", _abox, learning_rate=0.01),
    _pair("eike", "matrix-abox", _abox, learning_rate=0.01, instance_map="matrix"),
    _pair("concept2box", "abox", _abox, learning_rate=0.01),
]

EXEMPT = {}
