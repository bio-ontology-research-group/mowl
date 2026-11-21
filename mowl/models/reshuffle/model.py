r"""RESHUFFLE on a projected ontology."""
from mowl.models.graph_kge.region_kge import RegionKGEModel
from mowl.nn.graph.reshuffle import Reshuffle as ReshuffleKGE


class Reshuffle(RegionKGEModel):
    r"""RESHUFFLE [pavlovic2025]_ trained on the triples of a projection of the ontology.

    Entities carry random binary feature matrices, relations are soft coordinate
    reshuffling matrices, and a GNN propagates the features along the graph so that closed
    path rules are captured (see :mod:`mowl.nn.graph.reshuffle`). The default
    hyper-parameters are those of the best configurations in the authors' repository
    (``l=25``, ``k=80``, four layers, margin ranking loss with margin 2, 100 negatives,
    Adam with learning rate 0.005).

    Ontology link (an adaptation: RESHUFFLE is a knowledge graph method for closed path
    rules). The graph is a projection of the ontology (``subclassof``, existential, type
    and role assertion edges), and the model is trained on all of its edges, transductively.
    Note that the graph is encoded in full at every training step; this is meant for small
    and medium graphs. Scores exist for ``gci0`` and ``gci2``.

    :param l: Number of coordinates.
    :param k: Rows of an entity feature matrix.
    :param num_layers: Message passing layers.
    """

    KGE_CLASS = ReshuffleKGE

    def __init__(self, dataset, l=25,  # noqa: E741
                 k=80, num_layers=4, learning_rate=5e-3,
                 num_negs_per_pos=100, kge_kwargs=None, **kwargs):
        kge_kwargs = dict(kge_kwargs or {})
        kge_kwargs.update(l=l, k=k, num_layers=num_layers)
        kwargs.pop("embed_dim", None)
        super().__init__(dataset, learning_rate=learning_rate,
                         num_negs_per_pos=num_negs_per_pos, kge_kwargs=kge_kwargs, **kwargs)

    @staticmethod
    def _default_kge_kwargs(embed_dim):
        return {}
