r"""Box lattice embeddings (Vilnis et al., ACL 2018) as a module over the class taxonomy."""
import math

import torch as th
import torch.nn as nn

from mowl.baselines.nn.el.order_embeddings.base import TaxonomyModule


def meet(min_a, max_a, min_b, max_b):
    """Meet of two boxes: the intersection box ``(min, max)``. Components with
    ``max < min`` mean that the boxes are disjoint (the meet is the bottom element)."""
    return th.maximum(min_a, min_b), th.minimum(max_a, max_b)


def join(min_a, max_a, min_b, max_b):
    """Join of two boxes: the smallest enclosing box."""
    return th.minimum(min_a, min_b), th.maximum(max_a, max_b)


def volume(mn, mx):
    """Volume (product of side lengths) of a box, 0 if it is empty."""
    return th.relu(mx - mn).prod(dim=-1)


class UnitCubeBoxModule(TaxonomyModule):
    r"""Boxes in :math:`[0, 1]^d`, parametrised as (minimum, :math:`\Delta` = maximum
    minus minimum), as in the paper (Section 4.3) and its WordNet setup (Appendix D.1).

    The paper keeps the parameters in the cube and :math:`\Delta \ge \Delta_{\min}` by
    Euclidean projection after each gradient step; :meth:`project` is that projection and
    the model calls it after every step. ``class_embed`` holds ``[minimum, delta]``
    (``2 * embed_dim`` numbers per class).

    :param nb_ont_classes: Number of classes.
    :param embed_dim: Dimension :math:`d` of the cube.
    :param min_delta: :math:`\Delta_{\min}`. The paper tunes it in ``{0.01, ..., 1e-6}`` and reports
        ``1e-6`` for WordNet; the mOWL default is ``0.01``: with ``1e-6`` a dimension of a
        box can collapse to the floor during training (the log-probabilities then ignore it)
        and some stated pairs stay disjoint (about 1 seed in 12 on the tiny taxonomy test,
        0 of 60 with ``0.01``).
    :param init_side: Sides are initialised uniformly in ``[init_side[0], init_side[1]]``
        and minima so that the box lies in the cube.
    :param max_regularization: Weight of the L1 penalty :math:`\mathrm{mean}(1 - \max)`
        that pulls the upper corner to 1 and stops boxes from collapsing (Appendix D.1,
        "dimension-max regularization"; paper value 0.005).
    """

    def __init__(self, nb_ont_classes, embed_dim=50, min_delta=0.01, init_side=(0.3, 0.9),
                 max_regularization=0.0):
        super().__init__(nb_ont_classes)
        self.embed_dim = embed_dim
        self.min_delta = float(min_delta)
        self.max_regularization = float(max_regularization)
        self.class_embed = nn.Embedding(nb_ont_classes, 2 * embed_dim)
        with th.no_grad():
            side = th.empty(nb_ont_classes, embed_dim).uniform_(*init_side)
            low = th.rand(nb_ont_classes, embed_dim) * (1.0 - side)
            self.class_embed.weight.copy_(th.cat([low, side], dim=1))

    def corners(self, idx, detach_delta=False):
        """``(min, max)`` of the boxes ``idx``, shape ``(len(idx), d)``."""
        p = self.class_embed(idx)
        mn, delta = p[:, :self.embed_dim], p[:, self.embed_dim:]
        if detach_delta:
            delta = delta.detach()
        mn = mn.clamp(0.0, 1.0)
        delta = th.maximum(delta, th.full_like(delta, self.min_delta))
        return mn, mn + delta

    @th.no_grad()
    def project(self):
        """Project the parameters back onto the cube: ``min`` in ``[0, 1 - min_delta]``,
        ``delta`` in ``[min_delta, 1 - min]``."""
        w = self.class_embed.weight
        d = self.embed_dim
        mn = w[:, :d].clamp(0.0, 1.0 - self.min_delta)
        delta = th.minimum(w[:, d:].clamp(min=self.min_delta), 1.0 - mn)
        w.copy_(th.cat([mn, delta], dim=1))

    def regularization_loss(self):
        if self.max_regularization == 0.0:
            return 0
        _, mx = self.corners(th.arange(self.nb_ont_classes, device=self.class_embed.weight.device))
        return self.max_regularization * (1.0 - mx).mean()

    def marginal(self, idx):
        """Probability of each class: the volume of its box (uniform measure)."""
        mn, mx = self.corners(idx)
        return volume(mn.double(), mx.double())


class BoxLatticeModule(UnitCubeBoxModule):
    r"""Box lattice embeddings of the class taxonomy.

    **From the paper.** Concepts are boxes in the unit cube; box volume is probability
    under the uniform measure (Section 4.2), so :math:`P(D \mid C) = \mathrm{vol}(C \cap D)
    / \mathrm{vol}(C)`; the order is box inclusion, meet is the intersection (or
    :math:`\bot` if empty) and join the enclosing box (:func:`meet`, :func:`join`). The
    joint probability has no gradient for disjoint boxes, so the paper optimises the lower
    bound :math:`p(a) + p(b) - p(a \vee b) \le p(a \wedge b)` (Section 4.3), which is
    non-positive unless the boxes intersect; the true joint is used when they do. For
    WordNet the delta of the boxes of negative pairs is not updated (Appendix D.1).

    **mOWL.** :math:`C \sqsubseteq D` is the target :math:`P(D \mid C) = 1`, i.e.
    :math:`C \subseteq D`. With :math:`p` the joint estimate divided by
    :math:`\mathrm{vol}(C)` (lower bound if the boxes are disjoint, true value otherwise):

    - positive loss :math:`-\log(\max(p, 0) + \varepsilon) + w g`, with :math:`g =
      -(\text{lower bound}) / \mathrm{vol}(C \vee D) \in [0, 1)` for disjoint boxes and 0
      otherwise. For intersecting boxes this is the negative log conditional probability
      (the paper's cross entropy with target 1). The second term is *ours*: it keeps a
      gradient for disjoint boxes (the lower bound, normalised by the join so that it
      cannot blow up for tiny boxes), because the paper's log of the bound is
      not defined when the bound is non-positive (the sign of the log term printed in
      Section 4.3 is also inconsistent with its own text, so we did not copy it). The loss
      is zero exactly when :math:`C \subseteq D` (``zero_at_inclusion``).
    - negative loss :math:`-\log(1 - \min(\max(p, 0), 1 - \varepsilon))`, the cross entropy
      with target :math:`P(D \mid C) = 0`, computed with the delta of the boxes detached
      (``freeze_delta_negatives``).

    :param eps: :math:`\varepsilon`, the constant inside the logarithms.
    :param freeze_delta_negatives: Do not update delta through negative pairs (the paper's
        WordNet setting).
    :param disjoint_weight: Weight :math:`w` of :math:`g` in the positive loss. Default
        :math:`-\log\varepsilon`, so that the penalty of a disjoint stated pair has the same
        scale as the log term; with weight 1 its gradient is much smaller than that of the
        negatives and such a pair can stay disjoint.
    :param zero_at_inclusion: ``True`` (default): the log term is
        :math:`-\log((\min(\max(p, 0), 1) + \varepsilon) / (1 + \varepsilon))`, which is
        exactly 0 for an included pair (:math:`p = 1`) and never negative. ``False``: the
        log term of earlier versions, :math:`-\log(\max(p, 0) + \varepsilon)`, which is
        :math:`-\log(1 + \varepsilon) < 0` for an included pair. The two differ by the
        constant :math:`\log(1 + \varepsilon)`, so gradients are the same.
    """

    def __init__(self, nb_ont_classes, embed_dim=50, min_delta=0.01, init_side=(0.3, 0.9),
                 max_regularization=0.0, eps=1e-6, freeze_delta_negatives=True,
                 disjoint_weight=None, zero_at_inclusion=True):
        super().__init__(nb_ont_classes, embed_dim, min_delta, init_side, max_regularization)
        self.zero_at_inclusion = zero_at_inclusion
        self.eps = float(eps)
        self.disjoint_weight = -math.log(self.eps) if disjoint_weight is None \
            else float(disjoint_weight)
        self.freeze_delta_negatives = freeze_delta_negatives

    def _estimate(self, sub, sup, detach_delta=False):
        mn_a, mx_a = (t.double() for t in self.corners(sub, detach_delta))
        mn_b, mx_b = (t.double() for t in self.corners(sup, detach_delta))
        vol_a, vol_b = volume(mn_a, mx_a), volume(mn_b, mx_b)
        mn_m, mx_m = meet(mn_a, mx_a, mn_b, mx_b)
        mn_j, mx_j = join(mn_a, mx_a, mn_b, mx_b)
        intersect = (mx_m > mn_m).all(dim=-1)
        vol_j = volume(mn_j, mx_j)
        lower_bound = vol_a + vol_b - vol_j
        joint = th.where(intersect, volume(mn_m, mx_m), lower_bound)
        # fraction of the join that is not covered by the two boxes (0 if they intersect)
        gap = th.where(intersect, th.zeros_like(vol_j), -lower_bound / vol_j)
        return joint / vol_a, gap

    def conditional(self, sub, sup, detach_delta=False):
        r"""The estimate of :math:`P(\text{sup} \mid \text{sub})`: exact for intersecting
        boxes, the lower bound of Section 4.3 (``<= 0``) for disjoint ones."""
        return self._estimate(sub, sup, detach_delta)[0]

    def positive_loss(self, sub, sup):
        p, gap = self._estimate(sub, sup)
        if self.zero_at_inclusion:
            log_term = -th.log((p.clamp(min=0.0, max=1.0) + self.eps) / (1.0 + self.eps))
        else:
            log_term = -th.log(p.clamp(min=0.0) + self.eps)
        return (log_term + self.disjoint_weight * gap).float()

    def negative_loss(self, sub, sup):
        p = self.conditional(sub, sup, detach_delta=self.freeze_delta_negatives)
        p = p.clamp(min=0.0, max=1.0 - self.eps)
        return (-th.log1p(-p)).float()
