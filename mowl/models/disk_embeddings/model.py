from mowl.models.order_base import OrderEmbeddingModel
from mowl.nn.order.module import DiskEmbeddingsModule


class DiskEmbeddings(OrderEmbeddingModel):
    r"""Hyperbolic disk embeddings ([suzuki2019]_) for the subclass order of an ontology.

    The model is the combination of parts that the method ``disk_embeddings`` defines
    (:func:`mowl.representation.library.hierarchy_methods.disk_embeddings`): the class model is
    a disk (a point of the chosen space and a radius), the measure is the protrusion, the
    objective is the hierarchy objective; ``model.module`` is that
    :class:`~mowl.representation.ComposedGeometry`. The direct implementation it is tested
    against is ``mowl.baselines.models.disk_embeddings``.

    Each named class is a generalised formal disk :math:`(x, r)`: a centre in a hyperbolic
    (default), spherical or Euclidean space and a real radius that may be negative.
    :math:`C \sqsubseteq D` holds when the disk of :math:`D` contains that of :math:`C`,
    :math:`d(x_D, x_C) \le r_D - r_C`. The score is the protrusion
    :math:`l = d(x_D, x_C) - r_D + r_C` (Eq. 9), and the loss the margin loss of Eq. 10 with
    :math:`E = l` (Eq. 11): :math:`h_+(l)` for positives and :math:`h_+(\mu - l)` for
    negatives. Inclusion is transitive by the triangle inequality, and shifting all radii by
    a constant changes nothing (translational symmetry, Eq. 4). Optimisation is Riemannian
    SGD, with the exponential map of the Lorentz model for hyperbolic centres.

    Only subclass axioms between named classes (``gci0``) are used; see
    :class:`~mowl.models.order_base.OrderEmbeddingModel`.

    .. note::
        Adaptations. Ranking by the signed protrusion :math:`l` (not its positive part)
        avoids ties among all contained candidates. The paper does not state initialisation,
        learning rate or batch size for its WordNet experiments here; the spherical
        embeddings of the paper were initialised from Poincare embeddings, which is not done.
        Both radius and centre use the same step size (the paper's constants
        :math:`\lambda, \nu` of Eq. 12 are 1). The polyhedral quasi-metrics of Section 2.2,
        used to show the equivalence with order embeddings, are not implemented.

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension of the space (a hyperbolic point has ``embed_dim + 1``
        coordinates; spherical centres are unit vectors of ``R^embed_dim``).
    :param geometry: ``"hyperbolic"``, ``"spherical"`` or ``"euclidean"``.
    :param margin: :math:`\mu`.
    :param num_neg: Negatives per positive.
    """

    def __init__(self, dataset, embed_dim=10, geometry="hyperbolic", margin=0.01, num_neg=10,
                 learning_rate=0.1, batch_size=4096, model_filepath=None, device="cpu",
                 neg_sampling_gcis=None, seed=None, init_scale=0.1):
        super().__init__(dataset, embed_dim, learning_rate, batch_size, num_neg, device,
                         model_filepath, neg_sampling_gcis, seed)
        self.geometry = geometry
        self.margin = margin
        self.init_scale = init_scale
        self.init_module()

    def init_module(self):
        self.module = DiskEmbeddingsModule(
            len(self.class_index_dict), self.embed_dim, geometry=self.geometry,
            margin=self.margin, init_scale=self.init_scale, seed=self.seed).to(self.device)
        self.module.default_regimes = (self._default_loss,)
        self._attach_default_regimes()

    def _default_loss(self):
        """Eq. 10 and 11 of the paper: the hinge of the protrusion of each positive plus the
        mean cost of the negatives, the regime :class:`~mowl.training.SumLoss` with
        ``positive_floor``."""
        from mowl.training.losses import SumLoss
        return SumLoss(positive_floor=True)

    def compute_loss(self, pos_scores, neg_scores=None):
        """The loss of Eq. 10 and 11 on scores of positives and negatives: what the default
        regime (:meth:`_default_loss`) combines them to."""
        return self._default_loss().combine("gci0", pos_scores, neg_scores)
