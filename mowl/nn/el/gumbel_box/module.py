r"""Gumbel box embeddings (Dasgupta et al., NeurIPS 2020) over the class taxonomy, built from
parts."""
import torch as th

from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.hierarchy_methods import gumbel_box
from mowl.representation.library.hierarchy_modules import adopt


class GumbelBoxModule(ComposedGeometry):
    r"""Gumbel box embeddings of the class taxonomy, the combination that
    :func:`mowl.representation.library.hierarchy_methods.gumbel_box` names (read
    ``module.parts``): the box lattice's class model
    (:class:`~mowl.representation.library.hierarchy_classes.UnitCubeBoxes`), the smoothed
    volume ratio as the measure
    (:class:`~mowl.representation.library.hierarchy_measures.LogVolumeRatio` with the Gumbel
    scale :math:`\beta`) and the negative log probability of a stated pair, and
    :math:`-\log(1 - P)` for a corrupted pair, as the objective.

    **From the paper.** The lower corner of a box is a MaxGumbel and the upper corner a
    MinGumbel random variable with location parameters and common scale :math:`\beta`
    (Section 4). By Lemma 1 the intersection of boxes is again a Gumbel box; the expected
    side length is :math:`m(\mu_\vee - \mu_\wedge)` (Proposition 1), computed with the
    softplus approximation of Section 4.2, always positive, so disjoint boxes keep a
    gradient. The conditional probability is the ratio of two expected volumes (Section 4.1)
    and training is cross entropy against the target (Algorithm 1; as printed it has the
    signs of the two log-sum-exp terms swapped relative to Lemma 1, we follow Lemma 1).

    **mOWL.** :math:`C \sqsubseteq D` is the target :math:`P(D \mid C) = 1`; the cost is
    :math:`-\log P(D \mid C)`, that of a corrupted pair :math:`-\log(1 - P(D \mid C))`. At
    finite :math:`\beta` the cost is only zero in the limit :math:`\beta \to 0`.

    The earlier hand-written module is kept for comparison as
    ``mowl.baselines.nn.el.gumbel_box.GumbelBoxModule``.

    :param beta: Gumbel scale :math:`\beta` (paper: 0.01 to 1e-4 depending on the task).
    :param min_delta: Minimum side :math:`\Delta_{\min}` of the location boxes.
    :param max_regularization: Weight of the penalty pulling the upper corner to 1.
    :param log_guard: ``"stable"`` (default) or ``"epsilon"`` (earlier versions): the
        numerical guard of the logarithms; see
        :func:`~mowl.representation.library.hierarchy_methods.gumbel_box`.
    """

    def __init__(self, nb_ont_classes, embed_dim=50, beta=0.01, min_delta=1e-6,
                 init_side=(0.3, 0.9), max_regularization=0.0, log_guard="stable"):
        if beta <= 0:
            raise ValueError("beta must be positive")
        if log_guard not in ("stable", "epsilon"):
            raise ValueError("log_guard must be 'stable' or 'epsilon'.")
        adopt(self, gumbel_box(
            nb_ont_classes, embed_dim=embed_dim, beta=beta, min_delta=min_delta,
            init_side=init_side, max_regularization=max_regularization,
            log_guard=log_guard))
        self.nb_ont_classes = nb_ont_classes
        self.embed_dim = embed_dim
        self.beta = float(beta)
        self.min_delta = float(min_delta)
        self.max_regularization = float(max_regularization)
        self.log_guard = log_guard

    def corners(self, idx, detach_delta=False):
        """``(min, max)`` of the location boxes ``idx``, shape ``(len(idx), d)``."""
        region = self.class_model(idx, detach_size=detach_delta)
        return region.lower, region.upper

    def log_conditional(self, sub, sup, detach_delta=False):
        r""":math:`\log P(\text{sup} \mid \text{sub})` (always :math:`\le 0`), ``(B,)``."""
        a = self.class_model(sub, detach_size=detach_delta)
        b = self.class_model(sup, detach_size=detach_delta)
        return -self.measure.energy(a, b).reshape(-1)

    def positive_loss(self, sub, sup):
        """Cost of ``sub ⊑ sup``, ``(B,)``."""
        return self.inside(self.classes(sub), self.classes(sup)).reshape(-1)

    def negative_loss(self, sub, sup):
        """Cost of a pair that is *not* ordered, ``(B,)``."""
        return self.cost("gci0", th.stack([sub, sup], dim=1), True).reshape(-1)
