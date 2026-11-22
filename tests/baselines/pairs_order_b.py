"""Hyperbolic cones, shadow cones, disk embeddings, HiT and OnT: the combination of parts
(``mowl.models``) against the direct implementation (``mowl.baselines``)."""
import torch as th

from tests.baselines.harness import Pair

PAIRS = []
EXEMPT = {}

D = 6


# -- the three methods on manifolds (all pairs of named classes) --------------------------
def _pairs(model):
    n = len(model.named_class_index_dict)
    grid = th.cartesian_prod(th.arange(n), th.arange(n))
    return grid[grid[:, 0] != grid[:, 1]]


def _order_scores(model):
    """What the evaluation regime ranks: the energy of every pair of named classes (lower: more
    plausible), and the cost of the same pairs read as negatives."""
    rows = _pairs(model)
    with th.no_grad():
        return {"energy": model.module(rows, "gci0").reshape(-1),
                "negative": model.module(rows, "gci0", neg=True).reshape(-1)}


def _tables(model):
    module = model.module
    out = {"points": module.class_embed.weight}
    radius = getattr(module, "radius_embed", None)
    if radius is not None:
        out["radii"] = radius.weight
    return out


def _copy_order(old, new):
    new_tables, old_tables = _tables(new), _tables(old)
    with th.no_grad():
        for name, table in new_tables.items():
            table.copy_(old_tables[name])


def _load(side, name):
    """The class ``name`` as users get it (``new``) or as the direct implementation (``old``);
    imported when a pair is used, not when the file is collected."""
    import importlib
    module = {"new": "mowl.models.", "old": "mowl.baselines.models."}[side] + _MODULES[name]
    return getattr(importlib.import_module(module), name)


_MODULES = {"HyperbolicCones": "hyperbolic_cones", "ShadowCones": "shadow_cones",
            "DiskEmbeddings": "disk_embeddings", "HiT": "hit", "OnT": "ont"}


def _order_pair(key, name, **options):
    def build(side):
        return lambda dataset: _load(side, name)(dataset, embed_dim=D, num_neg=3, seed=0,
                                                 **options)
    return Pair(key, new=build("new"), old=build("old"), copy=_copy_order,
                scores=_order_scores, parameters=_tables, epochs=3)


def _order_pairs():
    pairs = [_order_pair("hyperbolic_cones", "HyperbolicCones", learning_rate=0.1)]
    for formulation in ("umbral-half-space", "penumbral-half-space", "umbral-ball",
                        "penumbral-ball"):
        pairs.append(_order_pair(f"shadow_cones/{formulation}", "ShadowCones",
                                 formulation=formulation, learning_rate=0.05,
                                 init_scale=0.4))
    pairs.append(_order_pair("shadow_cones/negative-energy", "ShadowCones",
                             negative="energy", learning_rate=0.05, init_scale=0.4))
    for space in ("hyperbolic", "spherical", "euclidean"):
        pairs.append(_order_pair(f"disk_embeddings/{space}", "DiskEmbeddings",
                                 geometry=space, learning_rate=0.1, margin=0.3))
    return pairs


# -- HiT and OnT (a text encoder and a Poincare ball) -------------------------------------
def _encoder():
    from mowl.nn.lm.encoders import BagOfWordsEncoder
    return BagOfWordsEncoder(dim=8, num_buckets=64)


_TEXT = dict(batch_size=8, warmup_steps=0, learning_rate=0.05, seed=0, centri_weight=0.5)


def _batch(model, size=24):
    """The first ``size`` training triplets, their texts and the embeddings of the negatives
    (the loss of one step)."""
    triplets = model.training_triplets()[:size]
    negatives = model.module.classes([t[2] for t in triplets]).centre \
        if hasattr(model.module, "classes") else model.module.embed([t[2] for t in triplets])
    return triplets, negatives


def _text_losses(model):
    triplets, negatives = _batch(model)
    loss, parts = model.batch_loss(triplets, negatives)
    return {"loss": loss, **{k: th.tensor(v) for k, v in parts.items()}}


def _text_scores(model):
    """The cost of every pair of classes by the measure (the ranking of the benchmarks)."""
    n = len(model.class_index_dict)
    rows = th.cartesian_prod(th.arange(n), th.arange(n))
    out = {"gci0": model.axiom_costs("gci0", rows)}
    if "gci2" in model.scored_forms and model.object_property_index_dict:
        r = len(model.object_property_index_dict)
        gci2 = th.cartesian_prod(th.arange(n), th.arange(r), th.arange(n))
        out["gci2"] = model.axiom_costs("gci2", gci2)
        out["gci3"] = model.axiom_costs("gci3", th.cartesian_prod(th.arange(r), th.arange(n),
                                                                  th.arange(n)))
    candidates = th.arange(n)
    out["ranking"] = model.score_candidates("gci0", rows[:12], 1, candidates)
    return out


def _encoder_of(model):
    module = model.module
    return module.class_model.encoder if hasattr(module, "class_model") else module.encoder


def _text_parameters(model):
    out = {f"encoder/{k}": v for k, v in _encoder_of(model).state_dict().items()}
    role = getattr(model.module, "role_model", None)
    if role is not None:
        out.update({f"role/{k}": v for k, v in role.state_dict().items()})
    return out


def _copy_text(old, new):
    with th.no_grad():
        _encoder_of(new).load_state_dict(_encoder_of(old).state_dict())
        role = getattr(old.module, "role_model", None)
        if role is not None:
            new.module.role_model.load_state_dict(role.state_dict())


def _text_train(model, epochs):
    model.train(epochs=epochs, validate=False)


def _text_pairs():
    def build(side, name, **options):
        return lambda dataset: _load(side, name)(dataset, encoder=_encoder(), **_TEXT,
                                                 **options)

    def pair(key, name, **options):
        return Pair(key, new=build("new", name, **options), old=build("old", name, **options),
                    copy=_copy_text, scores=_text_scores, losses=_text_losses,
                    train=_text_train, parameters=_text_parameters, epochs=3)

    return [pair("hit", "HiT", num_negatives=2),
            pair("hit/hard-negatives", "HiT", num_negatives=2, hard_negatives=True),
            pair("ont", "OnT"),
            pair("ont/no-roles", "OnT", use_roles=False),
            pair("ont/role-image", "OnT", existential_mode="role", role_scaling=True)]


PAIRS.extend(_order_pairs() + _text_pairs())
