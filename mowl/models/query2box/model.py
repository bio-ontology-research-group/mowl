"""Query2Box on ontologies."""
from mowl.models.query.model import QueryEmbeddingModel
from mowl.nn.query.geometry import Query2BoxGeometry


class Query2Box(QueryEmbeddingModel):
    """Query2Box as a query embedding model of an ontology.

    See :class:`~mowl.models.query.model.QueryEmbeddingModel` for the graph, the training
    and the class expression queries, and :class:`mowl.nn.query.geometry.Query2BoxGeometry`
    for the geometry. Extra keyword arguments go to the geometry.
    """

    geometry_class = Query2BoxGeometry
    default_gamma = 24.0
