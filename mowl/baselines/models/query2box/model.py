"""Query2Box on ontologies."""
from mowl.baselines.models.query.model import QueryEmbeddingModel
from mowl.baselines.nn.query.geometry import Query2BoxGeometry


class Query2Box(QueryEmbeddingModel):
    """Query2Box as a query embedding model of an ontology.

    See :class:`~mowl.baselines.models.query.model.QueryEmbeddingModel` for the graph, the training
    and the class expression queries, and :class:`mowl.baselines.nn.query.geometry.Query2BoxGeometry`
    for the geometry. Extra keyword arguments go to the geometry.
    """

    geometry_class = Query2BoxGeometry
    default_gamma = 24.0
