"""Region-based knowledge graph embeddings of a projected ontology.

:class:`RegionKGEModel` is :class:`~mowl.baselines.models.graph_kge.graph_pykeen_model.GraphPlusPyKEENModel`
with the projector, the PyKEEN model class, the optimiser and the training schedule chosen in
the constructor. BoxE, ExpressivE and RESHUFFLE are three subclasses that only name the
PyKEEN model class. Projection, training loop, negative sampling (PyKEEN's) and axiom scoring
are the ones of ``GraphPlusPyKEENModel``; nothing here is specific to one method.
"""

import torch as th

from mowl.baselines.models.graph_kge.graph_pykeen_model import GraphPlusPyKEENModel


class RegionKGEModel(GraphPlusPyKEENModel):
    """A PyKEEN knowledge graph embedding model trained on a projection of the ontology.

    :param dataset: mOWL dataset.
    :param embed_dim: Entity embedding dimension.
    :param projector: A projector of :mod:`mowl.projection`. Defaults to
        :class:`~mowl.projection.OWL2VecStarProjector`, whose graph has subclass edges,
        existential restriction edges, class assertion (type) edges and role assertion
        edges, so that class-level and individual-level triples are both used.
    :param learning_rate: Adam learning rate.
    :param batch_size: Training batch size.
    :param num_negs_per_pos: Negatives per positive triple (PyKEEN default if ``None``).
    :param seed: Random seed of the PyKEEN model.
    :param kge_kwargs: Further keyword arguments of the PyKEEN model class.
    """

    #: Constructor of the PyKEEN model; set by subclasses.
    KGE_CLASS = None

    def __init__(self, dataset, embed_dim=50, projector=None, learning_rate=1e-3,
                 batch_size=1024, num_negs_per_pos=None, seed=0, model_filepath=None,
                 device="cpu", kge_kwargs=None):
        super().__init__(dataset, model_filepath=model_filepath, device=device)
        if self.KGE_CLASS is None:
            raise TypeError("Use a subclass that sets KGE_CLASS (BoxE, ExpressivE, ...).")
        if projector is None:
            from mowl.projection import OWL2VecStarProjector
            projector = OWL2VecStarProjector()
        self.set_projector(projector)
        kwargs = dict(self._default_kge_kwargs(embed_dim))
        kwargs.update(kge_kwargs or {})
        self.set_kge_method(self.KGE_CLASS, random_seed=seed, **kwargs)
        self.optimizer = th.optim.Adam
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_negs_per_pos = num_negs_per_pos

    @staticmethod
    def _default_kge_kwargs(embed_dim):
        return dict(embedding_dim=embed_dim)
