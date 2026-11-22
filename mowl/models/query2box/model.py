"""Query2Box on ontologies."""
from mowl.models.query.model import QueryEmbeddingModel
from mowl.representation.library.query_methods import Query2BoxGeometry


class Query2Box(QueryEmbeddingModel):
    """Query2Box (Ren, Hu, Leskovec, ICLR 2020) as a query embedding model of an ontology.

    The combination is the builder :func:`mowl.representation.library.query_methods.query2box`:
    point entities, boxes of translations as relation model, the attention intersection of
    boxes, the out-of-box plus in-box distance, no negation. Any of those can be swapped
    through the extra keyword arguments (``intersection="exact"``, ``distance="l2"``,
    ``activation``, ``clamp_extent``, ``center_weight``).

    See :class:`~mowl.models.query.model.QueryEmbeddingModel` for the graph, the training
    and the class expression queries. The direct implementation of the paper is
    ``mowl.baselines.models.query2box.Query2Box``.
    """

    method = "query2box"
    geometry_class = Query2BoxGeometry
    default_gamma = 24.0
