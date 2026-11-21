r"""Box lattice embeddings of the class taxonomy."""
import torch as th

from mowl.models.order_embeddings.base import TaxonomyEmbeddingModel
from mowl.nn.el.box_lattice.module import BoxLatticeModule


class BoxLattice(TaxonomyEmbeddingModel):
    r"""Box lattice embeddings ([vilnis2018]_) of the subclass order between named classes.

    Classes are boxes in the unit cube, :math:`C \sqsubseteq D` is the conditional
    probability :math:`P(D \mid C) = \mathrm{vol}(C \cap D) / \mathrm{vol}(C) = 1`, trained
    with the cross entropy on conditional probabilities and the lower bound of the joint
    for disjoint boxes. See :class:`~mowl.nn.el.box_lattice.BoxLatticeModule` for what is
    from the paper and what is ours.

    **Supported.** Axioms :math:`C \sqsubseteq D` between named classes (training and
    scoring); the score of a candidate is the positive loss (lower is more plausible).
    The probabilistic reading :math:`P(D \mid C)` is available as
    ``model.module.conditional``. Conjunction, existentials, bottom and assertions are not
    used. **Adaptation**: the paper embeds WordNet and Flickr entailment with probabilities;
    here the target probabilities are 0 and 1 from stated and corrupted subclass pairs, plus
    optionally the unary marginals of the paper's WordNet setting.

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension of the cube (paper: 50).
    :param learning_rate: Adam learning rate (paper: 0.001).
    :param min_delta: Minimum side length (paper: 1e-6).
    :param unary_weight: Weight of the unary marginal loss: the cross entropy between the
        box volume of a class and the fraction of named classes that are its descendants in
        the training taxonomy (itself included), as in the paper's WordNet setup (weight 9.0
        there, edge weight 1.0). Defaults to ``0.0`` (not used).
    :param max_regularization: Weight of the dimension-max regularisation (paper 0.005).
        Defaults to ``0.0``.
    :param eps: Constant inside the logarithms.
    :param freeze_delta_negatives: Do not update delta through negatives (paper's WordNet
        setting).
    :param neg_columns: Columns to corrupt for negatives (default: both).
    """

    def __init__(self, dataset, embed_dim=50, learning_rate=0.001, min_delta=1e-6,
                 unary_weight=0.0, max_regularization=0.0, eps=1e-6,
                 freeze_delta_negatives=True, batch_size=4096, model_filepath=None,
                 device="cpu", neg_sampling_gcis=None, neg_columns=(0, 1)):
        super().__init__(dataset, embed_dim, batch_size, model_filepath=model_filepath,
                         device=device, learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis, neg_columns=neg_columns)
        self.min_delta = min_delta
        self.unary_weight = unary_weight
        self.max_regularization = max_regularization
        self.eps = eps
        self.freeze_delta_negatives = freeze_delta_negatives
        self._unary_target = None
        self.init_module()

    def init_module(self):
        self.module = BoxLatticeModule(
            len(self.class_index_dict), embed_dim=self.embed_dim, min_delta=self.min_delta,
            max_regularization=self.max_regularization, eps=self.eps,
            freeze_delta_negatives=self.freeze_delta_negatives).to(self.device)

    def additional_training_loss(self, epoch):
        if not self.unary_weight:
            return None
        if self._unary_target is None:
            target = self.descendant_fractions()
            self._unary_target = th.tensor(target, dtype=th.double, device=self.device)
        n = len(self._unary_target)
        p = self.module.marginal(th.arange(n, device=self.device)).clamp(1e-12, 1 - 1e-12)
        t = self._unary_target
        ce = -(t * th.log(p) + (1 - t) * th.log1p(-p))
        return self.unary_weight * ce.mean().float()
