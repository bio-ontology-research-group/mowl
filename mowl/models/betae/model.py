"""BetaE (Ren, Leskovec, NeurIPS 2020) on ontologies."""
from mowl.models.composed.query import QueryEmbeddingModel
from mowl.representation.library.query_methods import BetaEGeometry


class BetaE(QueryEmbeddingModel):
    """BetaE as a query embedding model of an ontology.

    The combination is the builder :func:`mowl.representation.library.query_methods.betae`:
    Beta distributions as entities, a network as relation model, the attention intersection,
    the inversion as negation and the KL divergence. Any of those can be swapped through the
    extra keyword arguments (``intersection="mean"``, ``distance="reverse kl"``,
    ``negation=None``, ``hidden``, ``layers``).

    See :class:`~mowl.models.composed.query.QueryEmbeddingModel` for the graph, the training
    and the class expression queries. The direct implementation of the paper is
    ``mowl.baselines.models.betae.BetaE``.
    """

    method = "betae"
    geometry_class = BetaEGeometry
    default_gamma = 24.0
