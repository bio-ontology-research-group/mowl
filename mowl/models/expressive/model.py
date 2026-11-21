r"""ExpressivE on a projected ontology."""
from mowl.models.graph_kge.region_kge import RegionKGEModel
from mowl.nn.graph.expressive import ExpressivE as ExpressivEKGE


class ExpressivE(RegionKGEModel):
    r"""ExpressivE [pavlovic2023]_ trained on the triples of a projection of the ontology.

    Entity pairs are points of the virtual triple space :math:`\mathbb{R}^{2d}` and every
    relation is a hyper-parallelogram there (see :mod:`mowl.nn.graph.expressive`). The
    geometry, the distance and the constraints are those of the authors' code; training
    and projection are the shared ones of
    :class:`~mowl.models.graph_kge.region_kge.RegionKGEModel`.

    Ontology link (an adaptation: ExpressivE is a knowledge graph method). As for
    :class:`~mowl.models.boxe.BoxE`, the triples are those of a projection of the ontology
    (``subclassof``, existential, type and role assertion edges). Composition patterns of
    the paper hold for the projected triples only; an ontology's role chains are not read.
    Scores exist for ``gci0`` and ``gci2``.

    :param min_denom: ``Dmin`` of the paper, lower bound of :math:`|1 - s^h s^t|`.
    :param p: Norm of the score.
    :param tanh_map: Map entities, centres and widths into ``[-1, 1]``.
    """

    KGE_CLASS = ExpressivEKGE

    def __init__(self, dataset, embed_dim=50, min_denom=0.5, p=2, tanh_map=True,
                 kge_kwargs=None, **kwargs):
        kge_kwargs = dict(kge_kwargs or {})
        kge_kwargs.update(min_denom=min_denom, p=p, tanh_map=tanh_map)
        super().__init__(dataset, embed_dim=embed_dim, kge_kwargs=kge_kwargs, **kwargs)
