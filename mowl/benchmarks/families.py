"""Runner support for the model families that are not normal-form EL models, and default
model factories for methods whose implementing class needs configuration.

A :class:`~mowl.benchmarks.runner.ModelFamily` tells the runner how to build and train a
model. :class:`ScoredModelFamily` covers every :class:`~mowl.base_models.Model` that
declares how it scores axioms (``axiom_scoring``, see
:mod:`mowl.evaluation.model_scoring`): ALC models, quantum-logic models, graph models.

Some catalogue methods are a configuration of a general class: DL2Vec is
``RandomWalkPlusW2VModel`` with the DL2Vec projector, a walker and Word2Vec. Their cards
name a *factory* (``MethodCard.factory``) that builds the configured model, so that
``run_benchmark(benchmark, method="dl2vec")`` works like for any other method. The
factories here are plain defaults for comparison runs, not the settings of a paper.
Every factory takes ``model_class``: the class it configures (default: the one of
:mod:`mowl.models`), so that another implementation of the same interface is configured
in exactly the same way.
"""

import inspect
import logging
import os

from mowl.benchmarks import runner

logger = logging.getLogger(__name__)


def _accepts(callable_, name):
    params = inspect.signature(callable_).parameters
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


class ScoredModelFamily(runner.ModelFamily):
    """Any :class:`~mowl.base_models.Model` subclass that declares ``axiom_scoring``.

    ``build`` passes ``device`` and ``model_filepath`` when the constructor takes them;
    ``train`` calls ``model.train`` with ``epochs`` (and ``validate_every`` when it is
    accepted) and keeps the parameters of the last epoch.
    """

    name = "scored"

    def matches(self, cls):
        from mowl.base_models.model import Model
        return (isinstance(cls, type) and issubclass(cls, Model)
                and getattr(cls, "axiom_scoring", None) is not None)

    def build(self, cls, dataset, model_kwargs, device, model_filepath):
        kwargs = dict(model_kwargs)
        if _accepts(cls.__init__, "device"):
            kwargs.setdefault("device", device)
        if _accepts(cls.__init__, "model_filepath"):
            kwargs.setdefault("model_filepath", model_filepath)
        return cls(dataset, **kwargs)

    def train(self, model, epochs, protocol, validate_every=None, select_best=True):
        kwargs = {}
        if validate_every is not None and _accepts(model.train, "validate_every"):
            kwargs["validate_every"] = validate_every
        if "epochs" in inspect.signature(model.train).parameters:
            model.train(epochs=epochs, **kwargs)
        else:
            model.train(epochs, **kwargs)
        return {"selected": "last epoch"}


# ---------------------------------------------------------------------------------------
# Default factories
# ---------------------------------------------------------------------------------------

def _walk_model(dataset, projector, embed_dim=100, num_walks=10, walk_length=20, alpha=0.1,
                window=5, min_count=1, workers=4, seed=0, model_filepath=None, device="cpu",
                walker="deepwalk", p=1.0, q=1.0, model_class=None):
    from mowl.walking import DeepWalk, Node2Vec
    if model_class is None:
        from mowl.models import RandomWalkPlusW2VModel as model_class
    model = model_class(dataset, model_filepath=model_filepath)
    model.set_projector(projector)
    outfile = (model_filepath or model.model_filepath) + ".walks"
    if walker == "node2vec":
        # mowl.walking.Node2Vec takes no seed (the Scala walker draws from an unseeded
        # generator): its walks differ from run to run, whatever ``seed`` says. Say so.
        logger.warning("node2vec walks are not seeded: the walks (and so the embeddings) "
                       "differ from run to run even with seed=%s; only Word2Vec follows "
                       "the seed.", seed)
        model.set_walker(Node2Vec(num_walks, walk_length, p=p, q=q, outfile=outfile,
                                  workers=workers))
    else:
        model.set_walker(DeepWalk(num_walks, walk_length, alpha, outfile=outfile,
                                  workers=workers, seed=seed))
    model.set_w2v_model(vector_size=embed_dim, window=window, min_count=min_count,
                        workers=workers, seed=seed, sg=1)
    model.device = device      # for the scoring head; Word2Vec itself runs on the CPU
    return model


def dl2vec(dataset, bidirectional_taxonomy=True, seed=0, **kwargs):
    """DL2Vec: DL2Vec projection, random walks, skip-gram Word2Vec."""
    from mowl.projection import DL2VecProjector
    return _walk_model(dataset, DL2VecProjector(bidirectional_taxonomy), seed=seed,
                       **kwargs)


def owl2vec_star(dataset, bidirectional_taxonomy=True, include_literals=False, seed=0,
                 **kwargs):
    """OWL2Vec*: its structural projection, random walks, skip-gram Word2Vec (no lexical
    document)."""
    from mowl.projection import OWL2VecStarProjector
    return _walk_model(dataset, OWL2VecStarProjector(
        bidirectional_taxonomy=bidirectional_taxonomy, include_literals=include_literals),
        seed=seed, **kwargs)


def taxonomy_walks(dataset, bidirectional_taxonomy=True, seed=0, **kwargs):
    """Random walks on the class hierarchy only, skip-gram Word2Vec."""
    from mowl.projection import TaxonomyProjector
    return _walk_model(dataset, TaxonomyProjector(bidirectional_taxonomy), seed=seed,
                       **kwargs)


def _corpus_model(dataset, with_annotations, embed_dim=100, window=5, min_count=1,
                  workers=4, seed=0, model_filepath=None, device="cpu", model_class=None):
    if model_class is None:
        from mowl.models import SyntacticPlusW2VModel as model_class
    corpus = (model_filepath + ".corpus") if model_filepath else None
    model = model_class(dataset, model_filepath=model_filepath, corpus_filepath=corpus)
    model.set_w2v_model(vector_size=embed_dim, window=window, min_count=min_count,
                        workers=workers, seed=seed, sg=1)
    model.generate_corpus(save=True, with_annotations=with_annotations)
    model.device = device      # for the scoring head; Word2Vec itself runs on the CPU
    return model


def onto2vec(dataset, seed=0, **kwargs):
    """Onto2Vec: Word2Vec on the axioms of the ontology written as sentences."""
    return _corpus_model(dataset, False, seed=seed, **kwargs)


def opa2vec(dataset, seed=0, **kwargs):
    """OPA2Vec: Word2Vec on the axioms and the annotations of the ontology."""
    return _corpus_model(dataset, True, seed=seed, **kwargs)


def graph_kge(dataset, projector="dl2vec", kge_method="TransE", embed_dim=100,
              learning_rate=1e-3, batch_size=1024, seed=0, model_filepath=None,
              device="cpu", model_class=None):
    """A knowledge graph embedding (PyKEEN) of a projection of the ontology.

    :param projector: ``"dl2vec"``, ``"owl2vec_star"``, ``"taxonomy"`` or a projector.
    :param kge_method: Name of a class of :mod:`pykeen.models` or the class.
    :param model_class: The class to configure (default
        :class:`mowl.models.GraphPlusPyKEENModel`); another implementation of the same
        interface can be configured the same way.
    """
    import pykeen.models
    import torch as th
    if model_class is None:
        from mowl.models import GraphPlusPyKEENModel as model_class
    from mowl.projection import DL2VecProjector, OWL2VecStarProjector, TaxonomyProjector
    if isinstance(projector, str):
        projector = {"dl2vec": lambda: DL2VecProjector(False),
                     "owl2vec_star": lambda: OWL2VecStarProjector(),
                     "taxonomy": lambda: TaxonomyProjector(False)}[projector]()
    if isinstance(kge_method, str):
        kge_method = getattr(pykeen.models, kge_method)
    model = model_class(dataset, model_filepath=model_filepath, device=device)
    model.set_projector(projector)
    model.set_kge_method(kge_method, embedding_dim=embed_dim, random_seed=seed)
    model.optimizer = th.optim.Adam
    model.lr = learning_rate
    model.batch_size = batch_size
    return model


def load_factory(path):
    """The function at the dotted ``path``."""
    import importlib
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def build_from_factory(card, dataset, model_kwargs, device, model_filepath):
    """Call the factory named by ``card.factory`` with the arguments it accepts."""
    factory = load_factory(card.factory)
    kwargs = dict(model_kwargs)
    kwargs.setdefault("device", device)
    kwargs.setdefault("model_filepath", model_filepath)
    os.makedirs(os.path.dirname(os.path.abspath(model_filepath)), exist_ok=True)
    return factory(dataset, **kwargs)
