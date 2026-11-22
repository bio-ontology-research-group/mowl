import torch as th

from mowl.baselines.models.order_base import OrderEmbeddingModel
from mowl.baselines.nn.order.module import ShadowConesModule


class ShadowCones(OrderEmbeddingModel):
    r"""Shadow cones ([yu2023shadow]_) for the subclass order of an ontology.

    Classes are points of the Poincare half-space or ball; :math:`C \sqsubseteq D` holds
    when the shadow that the object of :math:`C` casts lies inside the shadow cone of
    :math:`D` (a light source is at infinity or at the origin). ``formulation`` selects one
    of the four constructions of the paper: ``"umbral-half-space"`` (balls of radius
    ``radius`` attached to the points, light source a point at infinity; the best one in
    the paper), ``"penumbral-half-space"`` (horosphere light source), ``"umbral-ball"`` and
    ``"penumbral-ball"`` (light source at the origin of the Poincare ball, which leaves a
    hole of radius ``radius`` around it; ``"penumbral-ball"`` is the entailment cone of
    Ganea et al. with a different aperture parameter). The score of an axiom is the signed
    hyperbolic distance of the point of :math:`C` to the cone of :math:`D` (negative
    inside), and the loss is Eq. 1 of the paper,

    .. math::
        -\log \frac{\exp(-\max(E^+, \gamma_2))}{\sum_{k}\exp(\max(\gamma_1 - E^-_k, 0))}.

    Only subclass axioms between named classes (``gci0``) are used; see
    :class:`~mowl.baselines.models.order_base.OrderEmbeddingModel`.

    .. note::
        Adaptations and deviations. Curvature is fixed at :math:`-1` (the paper's
        experiments). Riemannian SGD is used for both models of space (the paper uses
        RiemannianAdam for the ball; it is not implemented, since ``geoopt``/HTorch are not
        dependencies); the half-space update is a retraction, not the exponential map. There
        is no burn-in phase. Training steps are full batches by default. For the ball
        formulations the closed form of the aperture of umbral cones is derived from the
        hyperbolic law of sines, :math:`\sin\beta = \sinh r/\sinh d_{\mathbb H}(u, O)`, and the
        altitude of :math:`u` as :math:`\mathrm{arcosh}(\cosh d_{\mathbb H}(u, O)/\cosh r)`,
        with the sign of
        the relative altitude reversed (it grows towards the light, as in the authors'
        code); the printed Appendix F.2 differs from this in what appear to be typographical
        errors, so umbral-ball is the least certain formulation (its transitivity is checked
        numerically in the tests). The signed distances of
        the penumbral-half-space formulation use the logarithm map of the half-space without
        evaluating :math:`\sinh s`.

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension of the hyperbolic space (5 or 10 in the paper).
    :param formulation: See above.
    :param radius: :math:`r` (0.05 for training in the paper; 0.1 for evaluation).
    :param horosphere_height: :math:`e^h` for ``"penumbral-half-space"`` (20 in the paper).
    :param gamma1: Distance :math:`\gamma_1` to which negatives are pushed out of the cone.
    :param gamma2: Distance :math:`\gamma_2` at which a positive is satisfied (0 in the paper).
    :param gamma3: Push of the apex into its cone for points of positive altitude (1e-4).
    :param negative: ``"hinge"`` (default): the module's negative loss is
        :math:`\max(0, \gamma_1 - E^-)`, as for every other model, so regimes that sum
        losses work; Eq. 1 is unchanged. ``"energy"``: the module returns the raw energy of
        a negative (earlier versions; not a loss). See
        :class:`~mowl.nn.order.module.ShadowConesModule`.
    :param num_neg: Negatives per positive (10 in the paper).
    """

    def __init__(self, dataset, embed_dim=10, formulation="umbral-half-space", radius=0.05,
                 horosphere_height=20.0, gamma1=0.01, gamma2=0.0, gamma3=1e-4, num_neg=10,
                 learning_rate=0.01, batch_size=4096, model_filepath=None, device="cpu",
                 neg_sampling_gcis=None, seed=None, init_scale=0.01, negative="hinge"):
        super().__init__(dataset, embed_dim, learning_rate, batch_size, num_neg, device,
                         model_filepath, neg_sampling_gcis, seed)
        self.formulation = formulation
        self.radius = radius
        self.horosphere_height = horosphere_height
        self.gamma1, self.gamma2, self.gamma3 = gamma1, gamma2, gamma3
        self.init_scale = init_scale
        self.negative = negative
        self.init_module()

    def init_module(self):
        self.module = ShadowConesModule(
            len(self.class_index_dict), self.embed_dim, formulation=self.formulation,
            radius=self.radius, horosphere_height=self.horosphere_height,
            gamma3=self.gamma3, init_scale=self.init_scale, seed=self.seed,
            margin=self.gamma1, negative=self.negative).to(self.device)

    def compute_loss(self, pos_scores, neg_scores=None):
        """Eq. 1 of the paper, averaged over the positives; ``neg_scores`` holds ``K``
        blocks of one value per positive: the hinge :math:`\\max(0, \\gamma_1 - E^-)`
        (``negative="hinge"``) or the raw energy :math:`E^-` (``"energy"``)."""
        pos = th.clamp(pos_scores.reshape(-1), min=self.gamma2)
        if neg_scores is None:
            return pos.mean()
        n = pos.shape[0]
        neg = neg_scores.reshape(-1, n).t()
        if self.negative == "energy":
            neg = th.relu(self.gamma1 - neg)
        return (pos + th.logsumexp(neg, dim=1)).mean()
