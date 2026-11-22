"""Comparisons of the group ``recomp-ball``: ELEmbeddings, EmEL++, EmEL(var) and RegD (boxes and
balls), built from parts (``mowl.models``) against the direct implementations
(``mowl.baselines.models``)."""
import itertools

import torch as th

from tests.baselines.harness import Pair

EMBED_DIM = 6


def _copy(old, new):
    """Every parameter of the direct module into the combination. The state dict of the direct
    module names its tables ``class_embed``, ``class_rad``, ...; the combination loads it
    under those names (and under the names of its parts)."""
    new.module.load_state_dict(old.module.state_dict())


def _rows(model, form, n=5):
    classes = range(min(n, len(model.class_index_dict)))
    relations = range(max(1, min(2, len(model.object_property_index_dict))))
    if form == "gci2":
        return th.tensor(list(itertools.product(classes, relations, classes)))
    return th.tensor(list(itertools.product(classes, classes)))


def _losses(model):
    """The loss of every axiom form on the training axioms, rows as the datasets list them
    (the same order for two models of one dataset)."""
    out = {}
    module = model.module
    for form, data in model.training_datasets.items():
        if len(data) == 0:
            continue
        rows = data[:]
        out[form] = module(rows, form)
        if form in getattr(module, "neg_capable_gcis", ()):
            out[form + "/neg"] = module(rows, form, neg=True)
    out.update(_random_rows_losses(model))
    return out


def _random_rows_losses(model):
    """The loss of every form of the module on random rows (the small dataset has no
    bottoms, and few conjunctions and assertions), positives and negatives."""
    module = model.module
    generator = th.Generator().manual_seed(7)
    n = len(model.class_index_dict)
    r = len(model.object_property_index_dict)
    i = len(model.individual_index_dict)

    def col(high, size=24):
        return th.randint(0, high, (size,), generator=generator)

    rows = {"gci0": [n, n], "gci1": [n, n, n], "gci2": [n, r, n], "gci3": [r, n, n],
            "gci0_bot": [n, n], "gci1_bot": [n, n], "gci3_bot": [r, n, n]}
    if i:
        rows.update({"class_assertion": [i, n], "object_property_assertion": [i, r, i]})
    if getattr(module, "role_axiom_capable", False):
        rows.update({"role_inclusion": [r, r], "role_chain": [r, r, r]})
    out = {}
    for form, highs in rows.items():
        data = th.stack([col(h) for h in highs], dim=1)
        out["random/" + form] = module(data, form)
        if form in module.neg_capable_gcis:
            out["random/" + form + "/neg"] = module(data, form, neg=True)
    return out


def _scores(*forms):
    """What the evaluation regime ranks: the model's own scoring of all pairs of a few classes
    for ``forms``."""
    def scores(model):
        out = {}
        for form in forms:
            model.eval_gci_name = form
            with th.no_grad():
                out[form] = model.eval_method(_rows(model, form))
        return out
    return scores


def _parameters(*names):
    def parameters(model):
        out = {}
        for name in names:
            table = getattr(model.module, name, None)
            if table is not None:
                out[name] = table.weight
        return out
    return parameters


BALL_TABLES = ("class_embed", "class_rad", "rel_embed", "ind_embed", "ind_rad")


def _pair(key, new_class, old_class, tables, forms=("gci2", "gci0"), **options):
    def build(cls):
        return lambda dataset: cls(dataset, embed_dim=EMBED_DIM, **options)
    return Pair(key, new=build(new_class), old=build(old_class), copy=_copy,
                scores=_scores(*forms), losses=_losses, parameters=_parameters(*tables))


def _pairs():
    from mowl.baselines.models.elembeddings.model import ELEmbeddings as OldELEm
    from mowl.baselines.models.emelpp.model import EmELPP as OldEmELPP, EmELVar as OldEmELVar
    from mowl.baselines.models.regd.model import RegDELBE as OldRegDELBE, RegDELEm as OldRegDELEm
    from mowl.models import ELEmbeddings, EmELPP, EmELVar, RegDELBE, RegDELEm

    box_tables = ("class_embed", "class_offset", "rel_embed", "ind_embed", "ind_offset")
    pairs = [
        _pair("elem", ELEmbeddings, OldELEm, BALL_TABLES, margin=0.1),
        _pair("elem/no_radius_term", ELEmbeddings, OldELEm, BALL_TABLES, margin=0.1,
              gci1_radius_term=False),
        _pair("emelpp", EmELPP, OldEmELPP, BALL_TABLES, margin=0.1),
        _pair("emelpp/role_axioms", EmELPP, OldEmELPP, BALL_TABLES, margin=0.1,
              load_role_axioms=True),
        _pair("emelvar", EmELVar, OldEmELVar, BALL_TABLES + ("rel_sigma",), margin=0.1,
              sigma_reg_weight=0.5),
        _pair("emelvar/paper_weight", EmELVar, OldEmELVar, BALL_TABLES + ("rel_sigma",),
              margin=0.1, load_role_axioms=True),
        _pair("regd", RegDELBE, OldRegDELBE, box_tables, margin=0.05, num_neg=2),
        _pair("regd/flip_gci2", RegDELBE, OldRegDELBE, box_tables, margin=0.05, num_neg=2,
              flip_gci2=True, depth_on_containment=False, lamda=0.5),
        _pair("regd_elem", RegDELEm, OldRegDELEm, BALL_TABLES, margin=0.05, num_neg=2),
        _pair("regd_elem/softplus", RegDELEm, OldRegDELEm, BALL_TABLES, margin=0.05,
              num_neg=2, radius="softplus", gci1_radius_term=False, p=1),
    ]
    return pairs


PAIRS = _pairs()

EXEMPT = {}
