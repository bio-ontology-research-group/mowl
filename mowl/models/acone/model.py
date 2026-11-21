"""AConE (He et al., 2024) on ontologies."""
from mowl.models.query.model import QueryEmbeddingModel
from mowl.nn.query.geometry import AConEGeometry


class AConE(QueryEmbeddingModel):
    """AConE as a query embedding model of an ontology.

    See :class:`~mowl.models.query.model.QueryEmbeddingModel` for the graph, the training
    and the class expression queries, and :class:`mowl.nn.query.geometry.AConEGeometry`
    for the geometry. Extra keyword arguments go to the geometry.
    """

    geometry_class = AConEGeometry
    default_gamma = 20.0
