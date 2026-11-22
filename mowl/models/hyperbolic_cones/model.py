from mowl.models.order_base import OrderEmbeddingModel
from mowl.nn.order.module import HyperbolicConesModule


class HyperbolicCones(OrderEmbeddingModel):
    r"""Hyperbolic entailment cones ([ganea2018]_) for the subclass order of an ontology.

    The model is the combination of parts that the method ``hyperbolic_cones`` defines
    (:func:`mowl.representation.library.hierarchy_methods.hyperbolic_cones`): the class model
    is a point of the Poincare ball with its geodesic cone, the measure is the angle against
    the aperture, the objective is the hierarchy objective (a hinge of margin ``margin`` for a
    pair that is not ordered); ``model.module`` is that
    :class:`~mowl.representation.ComposedGeometry`. The direct implementation it is tested
    against is ``mowl.baselines.models.hyperbolic_cones``.

    Each named class is a point :math:`x` of the Poincare ball with a cone of half-aperture
    :math:`\psi(x) = \arcsin(K(1 - \lVert x\rVert^2)/\lVert x\rVert)`; :math:`C \sqsubseteq D`
    holds when the point of :math:`C` lies in the cone of :math:`D`. The score of an axiom is
    the angle :math:`E(D, C) = \max(0, \Xi(D, C) - \psi(D))` that :math:`C` must be rotated
    around :math:`D` to enter the cone (0 when it is inside). The loss is the max-margin
    loss of Eq. 32, :math:`\sum E^+ + \sum \max(0, \gamma - E^-)` (here as means).

    The model uses the subclass axioms between named classes only (``gci0``); conjunctions,
    existential restrictions and roles are not represented and are ignored, and
    disjointness cannot be expressed by cones. Optimisation is Riemannian SGD with the
    exponential map of the Poincare ball and a projection that keeps points outside the
    :math:`\epsilon`-ball around the origin (:class:`~mowl.representation.library.manifolds.RiemannianSGD`).

    .. note::
        Adaptations. The paper initialises with 100 epochs of Poincare embeddings
        (Nickel and Kiela, with a burn-in phase) rescaled by 0.7; here points start at
        random directions with a norm uniform in :math:`[\epsilon, 0.5]` and there is no
        pre-training or burn-in. Training steps are full batches over the axioms (the paper
        uses batches of 10; attach :class:`~mowl.training.PerNormalFormMiniBatches` for
        mini-batches). Negatives are re-drawn when they are told axioms, but not when they
        are merely entailed (the paper's negatives are checked against the training
        edges). The Euclidean cones of the paper are not implemented.

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension of the Poincare ball (5 or 10 in the paper).
    :param K: Cone constant (0.1 in the paper).
    :param margin: :math:`\gamma` (0.01 in the paper).
    :param num_neg: Negatives per positive (10 in the paper).
    :param learning_rate: Step size of Riemannian SGD (the paper uses 1e-4 with summed
        losses on batches of 10; full-batch mean losses need a larger value).
    """

    def __init__(self, dataset, embed_dim=10, K=0.1, margin=0.01, num_neg=10,
                 learning_rate=0.1, batch_size=4096, model_filepath=None, device="cpu",
                 neg_sampling_gcis=None, seed=None, init_max_norm=0.5):
        super().__init__(dataset, embed_dim, learning_rate, batch_size, num_neg, device,
                         model_filepath, neg_sampling_gcis, seed)
        self.K = K
        self.margin = margin
        self.init_max_norm = init_max_norm
        self.init_module()

    def init_module(self):
        self.module = HyperbolicConesModule(
            len(self.class_index_dict), self.embed_dim, K=self.K, margin=self.margin,
            init_max_norm=self.init_max_norm, seed=self.seed).to(self.device)
