"""ConE (Zhang et al., NeurIPS 2021) on ontologies."""
from mowl.models.query.model import QueryEmbeddingModel
from mowl.representation.library.query_methods import ConEGeometry


class ConE(QueryEmbeddingModel):
    """ConE as a query embedding model of an ontology.

    The combination is the builder :func:`mowl.representation.library.query_methods.cone_qe`:
    sector cones as entities, a network as relation model, the attention intersection of
    cones, the complement as negation and the angular distance. Any of those can be swapped
    through the extra keyword arguments (``relations="rotation"``, ``signed_negation``,
    ``negation=None``, ``distance``, ``hidden``, ``layers``, ``dropout``, ``center_weight``).

    See :class:`~mowl.models.query.model.QueryEmbeddingModel` for the graph, the training
    and the class expression queries. The direct implementation of the paper is
    ``mowl.baselines.models.cone_qe.ConE``.
    """

    method = "cone_qe"
    geometry_class = ConEGeometry
    default_gamma = 20.0
