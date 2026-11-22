"""Knowledge graph models on a projection: the combination of parts against the direct code.

The direct implementations (``mowl.baselines.models``) train PyKEEN's models through
PyKEEN's training loop. The classes users get (``mowl.models``) are built from parts (the
projection, entities, relation model, interaction, objective, negative sampler) and trained by
the loop of ``TripleGraphModel``; the comparison shows that the two give the same losses,
scores and trained parameters.
"""
import contextlib

import torch as th

from tests.baselines.harness import Pair

LR, BATCH, EPOCHS = 0.02, 4, 3


def dataset():
    from tests.models.kg_region_data import abox_dataset
    return abox_dataset()


def _rep(rep):
    return rep(indices=None).detach()


def _assign(target, value):
    with th.no_grad():
        target.weight.copy_(value.reshape(target.weight.shape))


# -- the same input to both ---------------------------------------------------------------------
def axiom_scores(model):
    """The cost of every axiom a model scores, over all classes, roles and individuals."""
    classes = len(model.class_index_dict)
    roles = len(model.object_property_index_dict)
    individuals = len(model.individual_index_dict)
    cc = th.cartesian_prod(th.arange(classes), th.arange(classes))
    crc = th.cartesian_prod(th.arange(classes), th.arange(roles), th.arange(classes))
    ic = th.cartesian_prod(th.arange(individuals), th.arange(classes))
    iri = th.cartesian_prod(th.arange(individuals), th.arange(roles), th.arange(individuals))
    return {"gci0": model.axiom_costs("gci0", cc), "gci2": model.axiom_costs("gci2", crc),
            "class_assertion": model.axiom_costs("class_assertion", ic),
            "object_property_assertion": model.axiom_costs("object_property_assertion", iri)}


def triple_loss(model):
    """The loss of the whole training graph against one draw of corrupted triples, and the
    scores (higher: more plausible) of the positive and of the corrupted triples, each
    compared on its own."""
    from mowl.models.graph_kge.triple_model import TripleGraphModel
    factory = model.triples_factory
    positives = model.training_triples() if isinstance(model, TripleGraphModel) \
        else factory._add_inverse_triples_if_necessary(factory.mapped_triples)
    if isinstance(model, TripleGraphModel):
        module = model.module
        negatives = module.sampler(positives, factory.num_entities, factory.num_relations)
        pos = module.triple_cost(positives[:, 0], positives[:, 1], positives[:, 2])
        flat = negatives.reshape(-1, 3)
        neg = module.triple_cost(flat[:, 0], flat[:, 1], flat[:, 2]).reshape(
            *negatives.shape[:-1])
        loss = module.loss(-pos, -neg)
        pos, neg = -pos, -neg                # costs to scores
    else:
        from pykeen.sampling import BasicNegativeSampler
        kge = model.kge_method
        sampler = BasicNegativeSampler(
            mapped_triples=positives, num_entities=factory.num_entities,
            num_relations=factory.num_relations, num_negs_per_pos=model.num_negs_per_pos)
        negatives, _ = sampler.sample(positives)
        pos = kge.score_hrt(positives)
        neg = kge.score_hrt(negatives.reshape(-1, 3)).reshape(*negatives.shape[:-1])
        loss = kge.loss.process_slcwa_scores(
            positive_scores=pos, negative_scores=neg, label_smoothing=0.0, batch_filter=None,
            num_entities=factory.num_entities)
    return {"loss": loss, "positive scores": pos.reshape(-1),
            "negative scores": neg.reshape(-1)}


@contextlib.contextmanager
def _plain_pykeen_loop():
    """Run the direct implementation's PyKEEN training loop on the parameters it is given.

    PyKEEN's loop starts by drawing every parameter again (``reset_parameters_``, unless it
    continues a run), so the parameters a model starts with never decide the result and the
    two sides could not start from the same numbers; and it tries its batch once before it
    trains (``automatic memory optimization``, on by default, also on a CPU), which draws the
    parameters, batches and corrupted triples once more for nothing. The direct code is run as
    it is, with these two switched off: no draw of fresh parameters and no probe. Everything
    else (batches, corrupted triples, loss, optimiser, constraints) is PyKEEN's."""
    from pykeen.models.base import Model
    from pykeen.training import SLCWATrainingLoop

    from mowl.baselines.nn.graph.reshuffle import Reshuffle as ReshuffleKGE
    original_init = SLCWATrainingLoop.__init__
    original_reset = [(cls, cls.__dict__["reset_parameters_"]) for cls in (Model, ReshuffleKGE)]

    def init(self, *args, **kwargs):
        kwargs.setdefault("automatic_memory_optimization", False)
        original_init(self, *args, **kwargs)

    SLCWATrainingLoop.__init__ = init
    for cls, _ in original_reset:       # (RESHUFFLE draws its loop matrix in its own override)
        cls.reset_parameters_ = lambda self: self
    try:
        yield
    finally:
        SLCWATrainingLoop.__init__ = original_init
        for cls, function in original_reset:
            cls.reset_parameters_ = function


def train(model, epochs):
    from mowl.models.graph_kge.triple_model import TripleGraphModel
    model.lr, model.batch_size = LR, BATCH
    if isinstance(model, TripleGraphModel):
        model.train(epochs=epochs)
    else:
        with _plain_pykeen_loop():
            model.train(epochs=epochs)


# -- BoxE ------------------------------------------------------------------------------------
def _boxe_new(ds):
    from mowl.models import BoxE
    return BoxE(ds, embed_dim=6, seed=0)


def _boxe_old(ds):
    from mowl.baselines.models.boxe import BoxE
    return BoxE(ds, embed_dim=6, seed=0)


def _boxe_tables(old):
    kge = old.kge_method
    return {"entity": _rep(kge.entity_representations[0]),
            "bump": _rep(kge.entity_representations[1]),
            **{f"relation{i}": _rep(r) for i, r in enumerate(kge.relation_representations)}}


def _boxe_parameters(model):
    if hasattr(model, "module"):
        classes, relations = model.module.class_model, model.module.relation_model
        tables = [classes.centres, classes.bumps, relations.head_base, relations.head_delta,
                  relations.head_size, relations.tail_base, relations.tail_delta,
                  relations.tail_size]
        values = [t.weight.detach() for t in tables]
    else:
        kge = model.kge_method
        values = [_rep(r) for r in list(kge.entity_representations)
                  + list(kge.relation_representations)]
    names = ["entity", "bump", "head_base", "head_delta", "head_size", "tail_base",
             "tail_delta", "tail_size"]
    return {n: v.reshape(v.shape[0], -1) for n, v in zip(names, values)}


def _boxe_copy(old, new):
    values = _boxe_parameters(old)
    classes, relations = new.module.class_model, new.module.relation_model
    for target, name in ((classes.centres, "entity"), (classes.bumps, "bump"),
                         (relations.head_base, "head_base"),
                         (relations.head_delta, "head_delta"),
                         (relations.head_size, "head_size"),
                         (relations.tail_base, "tail_base"),
                         (relations.tail_delta, "tail_delta"),
                         (relations.tail_size, "tail_size")):
        _assign(target, values[name])


# -- ExpressivE ------------------------------------------------------------------------------
def _expressive_new(ds):
    from mowl.models import ExpressivE
    return ExpressivE(ds, embed_dim=6, seed=0)


def _expressive_old(ds):
    from mowl.baselines.models.expressive import ExpressivE
    return ExpressivE(ds, embed_dim=6, seed=0)


def _expressive_parameters(model):
    if hasattr(model, "module"):
        values = [model.module.class_model.centres.weight.detach(),
                  model.module.relation_model.vectors.weight.detach()]
    else:
        kge = model.kge_method
        values = [_rep(kge.entity_representations[0]), _rep(kge.relation_representations[0])]
    return dict(zip(("entity", "relation"), values))


def _expressive_copy(old, new):
    values = _expressive_parameters(old)
    _assign(new.module.class_model.centres, values["entity"])
    _assign(new.module.relation_model.vectors, values["relation"])


# -- RESHUFFLE -------------------------------------------------------------------------------
def _reshuffle_kwargs(**more):
    return dict(l=5, k=6, num_layers=2, num_negs_per_pos=4, **more)


def _reshuffle_new(**more):
    def build(ds):
        from mowl.models import Reshuffle
        return Reshuffle(ds, **_reshuffle_kwargs(**more))
    return build


def _reshuffle_old(**more):
    def build(ds):
        from mowl.baselines.models.reshuffle import Reshuffle
        return Reshuffle(ds, **_reshuffle_kwargs(**more))
    return build


def _reshuffle_parameters(model):
    if hasattr(model, "module"):
        relations = model.module.relation_model
        return {"relations": relations.vectors.weight.detach(),
                "loop": relations.loop_raw.detach()}
    kge = model.kge_method
    return {"relations": _rep(kge.relation_representations[0]),
            "loop": kge.loop_raw.detach()}


def _reshuffle_copy(old, new):
    values = _reshuffle_parameters(old)
    _assign(new.module.relation_model.vectors, values["relations"])
    with th.no_grad():
        new.module.relation_model.loop_raw.copy_(values["loop"])
    features = _rep(old.kge_method.entity_representations[0])
    assert th.equal(features.reshape(features.shape[0], -1),
                    new.module.class_model.centres.weight.detach())


def _flat(parameters):
    return lambda model: {k: v.reshape(v.shape[0], -1) if v.dim() > 1 else v
                          for k, v in parameters(model).items()}


# -- GraphPlusPyKEENModel (graph_kge, gda_projection) ----------------------------------------
def _graph_plus(model_class_path, projector, kge_name, **kge_options):
    def build(ds):
        import importlib

        import pykeen.models
        module_name, _, name = model_class_path.rpartition(".")
        cls = getattr(importlib.import_module(module_name), name)
        model = cls(ds)
        model.set_projector(projector())
        model.set_kge_method(getattr(pykeen.models, kge_name), embedding_dim=6, random_seed=0,
                             **kge_options)
        model.optimizer = th.optim.Adam
        model.lr, model.batch_size = LR, BATCH
        return model
    return build


NEW = "mowl.models.graph_kge.graph_pykeen_model.GraphPlusPyKEENModel"
OLD = "mowl.baselines.models.graph_kge.graph_pykeen_model.GraphPlusPyKEENModel"


def _dl2vec():
    from mowl.projection import DL2VecProjector
    return DL2VecProjector(False)


def _gda():
    from mowl.projection import GDAProjector
    return GDAProjector()


def _owl2vec():
    from mowl.projection import OWL2VecStarProjector
    return OWL2VecStarProjector()


def _graph_copy(old, new):
    new.kge_method.load_state_dict(old.kge_method.state_dict())


def _graph_parameters(model):
    return {k: v.detach() for k, v in model.kge_method.state_dict().items()}


def _graph_pair(key, projector, kge_name):
    return Pair(key=key, new=_graph_plus(NEW, projector, kge_name),
                old=_graph_plus(OLD, projector, kge_name), copy=_graph_copy,
                scores=axiom_scores, losses=triple_loss, train=train,
                parameters=_graph_parameters, dataset=dataset)


PAIRS = [
    Pair(key="boxe", new=_boxe_new, old=_boxe_old, copy=_boxe_copy, scores=axiom_scores,
         losses=triple_loss, train=train, parameters=_boxe_parameters, dataset=dataset),
    Pair(key="expressive", new=_expressive_new, old=_expressive_old, copy=_expressive_copy,
         scores=axiom_scores, losses=triple_loss, train=train,
         parameters=_expressive_parameters, dataset=dataset),
    Pair(key="reshuffle", new=_reshuffle_new(), old=_reshuffle_old(), copy=_reshuffle_copy,
         scores=axiom_scores, losses=triple_loss, train=train,
         parameters=_flat(_reshuffle_parameters), dataset=dataset),
    Pair(key="reshuffle/forward-triples", new=_reshuffle_new(inverse_triples=False),
         old=_reshuffle_old(inverse_triples=False), copy=_reshuffle_copy,
         scores=axiom_scores, losses=triple_loss, train=train,
         parameters=_flat(_reshuffle_parameters), dataset=dataset),
    _graph_pair("graph_kge", _dl2vec, "TransE"),
    _graph_pair("graph_kge/distmult", _owl2vec, "DistMult"),
    _graph_pair("gda_projection", _gda, "TransE"),
]
EXEMPT = {
    "composed_graph_kge": "GraphKGEModel is the generic combination of parts for knowledge "
                          "graph embeddings of a projection; it has no direct implementation "
                          "to be compared with (its named combinations graph_kge, "
                          "gda_projection, boxe, expressive and reshuffle are compared "
                          "above).",
}
