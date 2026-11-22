"""Query2Box, BetaE, ConE and AConE: the combination of parts against the direct model.

The combination is ``mowl.models.<Method>`` (a ``ComposedQueryGeometry`` built from parts, the
sampler and the objective of ``mowl.models.query``); the direct implementation is
``mowl.baselines.models.<m>`` with its own hand-written geometry and loss.
"""
import torch as th

from tests.baselines.harness import Pair

_STEPS = 3
_COMMON = dict(embed_dim=8, learning_rate=3e-3, batch_size=24, negatives=4, seed=0,
               steps_per_epoch=_STEPS)


def _builder(key):
    from mowl.representation.library import query_methods as qm
    return {"query2box": qm.query2box, "betae": qm.betae, "cone_qe": qm.cone_qe,
            "acone": qm.acone}[key]


def _new(cls_name, module, **options):  # noqa: ARG001
    def build(dataset):
        import importlib
        cls = getattr(importlib.import_module("mowl.models"), cls_name)
        return cls(dataset, **_COMMON, **options)
    return build


def _old(cls_name, module, **options):
    def build(dataset):
        import importlib
        cls = getattr(importlib.import_module(f"mowl.baselines.models.{module}"), cls_name)
        return cls(dataset, **_COMMON, **options)
    return build


def _copy(key):
    def copy(old, new):
        # the combination that shares the parameters of the direct module has the same
        # parts as the new module, so the state dict maps one to one
        shared = _builder(key)(tables=old.module)
        new.module.load_state_dict(shared.state_dict(), strict=True)
    return copy


def _scores(model):
    """What the evaluation regimes rank: the scores of queries of several structures against
    all entities, and the cost of the axiom forms the model scores."""
    n = model.num_classes
    out = {}
    structures = {
        "1p": (("e", ("r",)), [(0, (1,)), (2, (1,))]),
        "2p": (("e", ("r", "r")), [(0, (1, 3)), (1, (1, 1))]),
        "2i": ((("e", ("r",)), ("e", ("r",))), [((0, (1,)), (2, (1,)))]),
    }
    if model.module.supports_negation:
        structures["2in"] = ((("e", ("r",)), ("e", ("r", "n"))),
                             [((0, (1,)), (1, (1, -2)))])
        structures["3in"] = ((("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))),
                             [((0, (1,)), (2, (1,)), (1, (1, -2)))])
    for name, (structure, queries) in structures.items():
        out["query/" + name] = model.score_query(structure, queries)
    classes = th.arange(n)
    pairs = th.cartesian_prod(classes[:6], classes[:6])
    out["gci0"] = model.axiom_costs("gci0", pairs)
    roles = len(model._role_ids)
    if roles:
        rows = th.stack([th.arange(5) % n, th.zeros(5, dtype=th.long),
                         (th.arange(5) + 1) % n], 1)
        out["gci2"] = model.axiom_costs("gci2", rows)
    if model._individual_ids:
        rows = th.stack([th.arange(4) % len(model._individual_ids), th.arange(4) % n], 1)
        out["class_assertion"] = model.axiom_costs("class_assertion", rows)
    return out


def _losses(model):
    batch = model._batch() if hasattr(model, "_batch") else model.sampler.batch()
    return {"loss": model.loss(batch)}


def _parameters(model):
    return {"entities": model.module.entity_vectors(),
            "relations": model.module.relation_vectors()}


def _train(model, epochs):
    model.train(epochs=epochs)


def _pair(key, cls_name, module, variant=None, **options):
    return Pair(key=key + (f"/{variant}" if variant else ""),
                new=_new(cls_name, module, **options), old=_old(cls_name, module, **options),
                copy=_copy(key), scores=_scores, losses=_losses, train=_train,
                parameters=_parameters)


def _released_dataset():
    """The family dataset with a few released training queries (1p and 2i structures), so that
    both the path and the other group of the released sampler are drawn from."""
    import copy
    from mowl.datasets.builtin.query_sets import ReleasedTrainingQueries
    from tests.baselines.harness import default_dataset
    stand_in = copy.copy(default_dataset())
    one = ("e", ("r",))

    def training_queries(entity_id, relation_id):
        return {
            "1p": ReleasedTrainingQueries(one, [[0, 1], [1, 1], [2, 1]], [0, 1, 3, 7],
                                          [1, 2, 3, 4, 5, 6, 0]),
            "2i": ReleasedTrainingQueries((one, one), [[0, 1, 2, 1], [1, 1, 3, 1]], [0, 2, 3],
                                          [4, 5, 6])}
    stand_in.training_queries = training_queries
    return stand_in


def _released_losses(model):
    """The weighted loss of one path batch and one other batch of the released sampler: the
    combination through ``model.loss`` of the batch of its sampler, the direct model through
    its own ``released_loss`` of its own batch."""
    if hasattr(model, "_released_batch") and not hasattr(model, "sampler"):
        return {g: model.released_loss(model._released_batch(g)) for g in ("path", "other")}
    return {g: model.loss(model.sampler.batch(g)) for g in ("path", "other")}


PAIRS = [
    _pair("query2box", "Query2Box", "query2box"),
    _pair("query2box", "Query2Box", "query2box", "softplus-clamped", activation="softplus",
          clamp_extent=True),
    _pair("query2box", "Query2Box", "query2box", "no-extension", extension=False,
          batch_per_structure=True),
    _pair("betae", "BetaE", "betae"),
    _pair("betae", "BetaE", "betae", "two-layers", layers=2, hidden=16),
    _pair("cone_qe", "ConE", "cone_qe"),
    _pair("cone_qe", "ConE", "cone_qe", "signed-negation", signed_negation=True),
    _pair("acone", "AConE", "acone"),
    _pair("acone", "AConE", "acone", "authors-readings", signed_negation=True, aperture="wrap"),
    _pair("acone", "AConE", "acone", "clipped", aperture="clip"),
]
PAIRS.append(Pair(
    key="query2box/released", new=_new("Query2Box", "query2box", queries="released"),
    old=_old("Query2Box", "query2box", queries="released"), copy=_copy("query2box"),
    scores=_scores, losses=_released_losses, train=_train, parameters=_parameters,
    dataset=_released_dataset))

EXEMPT = {}
