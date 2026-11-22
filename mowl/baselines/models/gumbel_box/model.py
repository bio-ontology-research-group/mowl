r"""Gumbel box embeddings of the class taxonomy."""
from mowl.models.order_parts_a import TaxonomyEmbeddingModel
from mowl.baselines.nn.el.gumbel_box.module import GumbelBoxModule


class GumbelBox(TaxonomyEmbeddingModel):
    r"""Gumbel box embeddings ([dasgupta2020]_) of the subclass order between named classes.

    Like :class:`~mowl.models.box_lattice.BoxLattice` with box corners that are Gumbel
    random variables: the intersection is again a Gumbel box and the softplus-smoothed
    expected volume gives a gradient also for disjoint boxes, so no lower bound trick is
    needed. See :class:`~mowl.nn.el.gumbel_box.GumbelBoxModule` for what is from the paper
    and what is ours.

    **Supported.** Axioms :math:`C \sqsubseteq D` between named classes (training and
    scoring; the score is :math:`-\log P(D \mid C)`, lower is more plausible).
    **Adaptation**: the paper targets conditional probabilities of tree and graph
    hierarchies, WordNet-style, and MovieLens; here the targets are 1 for stated subclass
    pairs and 0 for corrupted pairs.

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension of the cube.
    :param beta: Gumbel scale (the paper's ranges: 0.001 to 3; 0.01 for MovieLens).
    :param learning_rate: Adam learning rate.
    :param min_delta: Minimum side length of the location boxes.
    :param max_regularization: Weight of the penalty pulling the upper location corner to 1
        (``0.0``: none).
    :param log_guard: ``"stable"`` (default) or ``"epsilon"`` (earlier versions): the
        numerical guard of the logarithms, see
        :class:`~mowl.nn.el.gumbel_box.GumbelBoxModule`.
    :param neg_columns: Columns to corrupt for negatives (default: both).
    """

    def __init__(self, dataset, embed_dim=50, beta=0.01, learning_rate=0.001,
                 min_delta=1e-6, max_regularization=0.0, batch_size=4096,
                 model_filepath=None, device="cpu", neg_sampling_gcis=None,
                 neg_columns=(0, 1), log_guard="stable"):
        super().__init__(dataset, embed_dim, batch_size, model_filepath=model_filepath,
                         device=device, learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis, neg_columns=neg_columns)
        self.beta = beta
        self.min_delta = min_delta
        self.max_regularization = max_regularization
        self.log_guard = log_guard
        self.init_module()

    def init_module(self):
        self.module = GumbelBoxModule(
            len(self.class_index_dict), embed_dim=self.embed_dim, beta=self.beta,
            min_delta=self.min_delta,
            max_regularization=self.max_regularization,
            log_guard=self.log_guard).to(self.device)
