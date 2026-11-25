"""BetaE (Ren, Leskovec, NeurIPS 2020) on ontologies."""
from mowl.baselines.models.query.model import QueryEmbeddingModel
from mowl.baselines.nn.query.geometry import BetaEGeometry


class BetaE(QueryEmbeddingModel):
    """BetaE as a query embedding model of an ontology.

    See :class:`~mowl.baselines.models.query.model.QueryEmbeddingModel` for the graph, the training
    and the class expression queries, and :class:`mowl.baselines.nn.query.geometry.BetaEGeometry`
    for the geometry. Extra keyword arguments go to the geometry.
    """

    geometry_class = BetaEGeometry
    default_gamma = 24.0
