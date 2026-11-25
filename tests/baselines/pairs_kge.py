"""``mowl.kge.KGEModel`` (released in v2.1.0) against its released code path.

The direct implementation (``mowl.baselines.kge``) is the class as released: a PyKEEN model
trained by PyKEEN's ``SLCWATrainingLoop``. The class users get trains in mOWL's shared loop
with the regime ``SLCWABatches``, which reads PyKEEN's batches, negatives and loss; the
comparison shows that the run is the released one (same losses, scores and parameters after
training). Both sides wrap the same kind of PyKEEN model, so there is no geometry of parts to
check (``parts`` is skipped): what is compared is the training run.
"""
import os

import torch as th

from tests.baselines.harness import Pair


def _factory():
    from mowl.projection import Edge
    edges = [Edge("a", "r", "b"), Edge("b", "r", "c"), Edge("c", "s", "a"),
             Edge("a", "s", "d"), Edge("d", "r", "c"), Edge("e", "r", "a"),
             Edge("b", "s", "e")]
    return Edge.as_pykeen(edges)


def _build(cls_path):
    def build(factory):
        import importlib
        from pykeen.models import TransE
        module, name = cls_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module), name)
        model = TransE(triples_factory=factory, embedding_dim=4, random_seed=0)
        path = os.path.join(os.environ.get("TMPDIR", "."), f"kge_pair_{module}.pt")
        return cls(factory, model, 3, batch_size=3, model_filepath=path)
    return build


def _copy(old, new):
    new.model.load_state_dict(old.model.state_dict())


def _scores(model):
    triples = model.triples_factory.mapped_triples
    with th.no_grad():
        return {"hrt": model.score_method_tensor(triples).reshape(-1)}


def _losses(model):
    """The model's loss of the training triples against fixed corruptions of their tails."""
    kge = model.model
    triples = model.triples_factory.mapped_triples
    corrupted = triples.clone()
    corrupted[:, 2] = (corrupted[:, 2] + 1) % model.triples_factory.num_entities
    pos, neg = kge.score_hrt(triples), kge.score_hrt(corrupted).view(-1, 1)
    return {"loss": kge.loss.process_slcwa_scores(positive_scores=pos, negative_scores=neg),
            "regularization": th.as_tensor(kge.collect_regularization_term())}


def _parameters(model):
    kge = model.model
    return {"entities": kge.entity_representations[0](indices=None).detach(),
            "relations": kge.relation_representations[0](indices=None).detach()}


def _train(model, epochs):
    model.epochs = epochs
    model.train()


PAIRS = [Pair(key="kge_model", new=_build("mowl.kge.KGEModel"),
              old=_build("mowl.baselines.kge.KGEModel"), copy=_copy, scores=_scores,
              losses=_losses, train=_train, parameters=_parameters, dataset=_factory,
              epochs=3, atol=0.0, rtol=0.0,
              skip={"parts": "KGEModel wraps a PyKEEN model given by the user (no geometry "
                             "of parts); the comparison is of its training run"})]
EXEMPT = {}
