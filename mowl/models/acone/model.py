"""AConE (He et al., 2024) on ontologies."""
from mowl.models.query.model import QueryEmbeddingModel
from mowl.representation.library.query_methods import AConEGeometry


class AConE(QueryEmbeddingModel):
    """AConE as a query embedding model of an ontology.

    The combination is the builder :func:`mowl.representation.library.query_methods.acone`:
    ConE with the rotation as relation model. Any part can be swapped through the extra
    keyword arguments (``aperture``, ``signed_negation``, ``negation=None``,
    ``intersection``, ``distance``, ``dropout``, ``center_weight``).

    ``signed_negation`` defaults to ``False`` here (as in earlier versions of this class: the
    complement ``pi - |a|``), while the builder of the method has the authors' reading
    (``True``, ``pi - a``), which :mod:`mowl.benchmarks` selects for the published recipes.

    See :class:`~mowl.models.query.model.QueryEmbeddingModel` for the graph, the training
    and the class expression queries. The direct implementation of the paper is
    ``mowl.baselines.models.acone.AConE``.
    """

    method = "acone"
    geometry_class = AConEGeometry
    default_gamma = 20.0
    geometry_defaults = {"signed_negation": False}
