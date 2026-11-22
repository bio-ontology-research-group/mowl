"""ConE (Zhang et al., NeurIPS 2021) on ontologies."""
from mowl.baselines.models.query.model import QueryEmbeddingModel
from mowl.baselines.nn.query.geometry import ConEGeometry


class ConE(QueryEmbeddingModel):
    """ConE as a query embedding model of an ontology.

    See :class:`~mowl.baselines.models.query.model.QueryEmbeddingModel` for the graph, the training
    and the class expression queries, and :class:`mowl.baselines.nn.query.geometry.ConEGeometry`
    for the geometry. Extra keyword arguments go to the geometry.
    """

    geometry_class = ConEGeometry
    default_gamma = 20.0
