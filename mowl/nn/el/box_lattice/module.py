r"""Box lattice embeddings (Vilnis et al., ACL 2018) over the class taxonomy, built from
parts."""
import torch as th

from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.hierarchy_methods import box_lattice
from mowl.representation.library.hierarchy_modules import FlatLossOption, adopt


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


class BoxLatticeModule(FlatLossOption, ComposedGeometry):
    r"""Box lattice embeddings of the class taxonomy, the combination that
    :func:`mowl.representation.library.hierarchy_methods.box_lattice` names (read
    ``module.parts``): boxes in the unit cube
    (:class:`~mowl.representation.library.hierarchy_classes.UnitCubeBoxes`, parametrised as
    minimum and :math:`\Delta` = maximum minus minimum, kept in the cube by ``project()``),
    the volume ratio as the measure
    (:class:`~mowl.representation.library.hierarchy_measures.LogVolumeRatio`), the cross
    entropy with target 1 for a stated pair and target 0 for a corrupted pair as the
    objective, and the "dimension-max regularisation" as an optional regulariser.

    **From the paper.** Concepts are boxes in the unit cube; box volume is probability under
    the uniform measure (Section 4.2), so :math:`P(D \mid C) = \mathrm{vol}(C \cap D) /
    \mathrm{vol}(C)`; the order is box inclusion. The joint probability has no gradient for
    disjoint boxes, so the paper optimises the lower bound :math:`p(a) + p(b) - p(a \vee b)
    \le p(a \wedge b)` (Section 4.3). For WordNet the delta of the boxes of negative pairs
    is not updated (Appendix D.1).

    **mOWL.** :math:`C \sqsubseteq D` is the target :math:`P(D \mid C) = 1`. With :math:`p`
    the estimate (lower bound if the boxes are disjoint, true value otherwise):

    - cost of a stated pair :math:`-\log((\min(\max(p, 0), 1) + \varepsilon) / (1 +
      \varepsilon)) + w g`, with :math:`g = -(\text{lower bound}) / \mathrm{vol}(C \vee D)`
      for disjoint boxes and 0 otherwise (a term of mOWL that keeps a gradient for a stated
      pair that is disjoint); exactly 0 for an included pair (``zero_at_inclusion``);
    - cost of a corrupted pair :math:`-\log(1 - \min(\max(p, 0), 1 - \varepsilon))`, with
      the delta of the boxes detached (``freeze_delta_negatives``).

    The earlier hand-written module is kept for comparison as
    ``mowl.baselines.nn.el.box_lattice.BoxLatticeModule``.

    :param min_delta: :math:`\Delta_{\min}` (paper: 1e-6 for WordNet; default 0.01: with
        1e-6 a dimension can collapse during training and some stated pairs stay disjoint).
    :param init_side: Sides are initialised uniformly in this range.
    :param max_regularization: Weight of :math:`\mathrm{mean}(1 - \max)` (paper 0.005).
    :param eps: :math:`\varepsilon`, the constant inside the logarithms.
    :param freeze_delta_negatives: Do not update delta through negative pairs.
    :param disjoint_weight: Weight :math:`w` of :math:`g`; default :math:`-\log\varepsilon`.
    :param zero_at_inclusion: ``False`` is the earlier guard,
        :math:`-\log(1 + \varepsilon)` for an included pair.
    """

    def __init__(self, nb_ont_classes, embed_dim=50, min_delta=0.01, init_side=(0.3, 0.9),
                 max_regularization=0.0, eps=1e-6, freeze_delta_negatives=True,
                 disjoint_weight=None, zero_at_inclusion=True):
        adopt(self, box_lattice(
            nb_ont_classes, embed_dim=embed_dim, min_delta=min_delta, init_side=init_side,
            max_regularization=max_regularization, eps=eps,
            freeze_delta_negatives=freeze_delta_negatives, disjoint_weight=disjoint_weight,
            zero_at_inclusion=zero_at_inclusion))
        self.nb_ont_classes = nb_ont_classes
        self.embed_dim = embed_dim
        self.min_delta = float(min_delta)
        self.max_regularization = float(max_regularization)
        self.eps = self.measure.eps
        self.disjoint_weight = self.measure.disjoint_weight
        self.freeze_delta_negatives = freeze_delta_negatives
        self.zero_at_inclusion = zero_at_inclusion

    def corners(self, idx, detach_delta=False):
        """``(min, max)`` of the boxes ``idx``, shape ``(len(idx), d)``."""
        region = self.class_model(idx, detach_size=detach_delta)
        return region.lower, region.upper

    def marginal(self, idx):
        """Probability of each class: the volume of its box (uniform measure)."""
        mn, mx = self.corners(idx)
        return volume(mn.double(), mx.double())

    def conditional(self, sub, sup, detach_delta=False):
        r"""The estimate of :math:`P(\text{sup} \mid \text{sub})`, ``(B,)``: exact for
        intersecting boxes, the lower bound of Section 4.3 (``<= 0``) for disjoint ones."""
        a = self.class_model(sub, detach_size=detach_delta)
        b = self.class_model(sup, detach_size=detach_delta)
        return self.measure.conditional(a, b)[0].reshape(-1)

    def positive_loss(self, sub, sup):
        """Cost of ``sub ⊑ sup``, ``(B,)``."""
        return self.inside(self.classes(sub), self.classes(sup)).reshape(-1)

    def negative_loss(self, sub, sup):
        """Cost of a pair that is *not* ordered, ``(B,)``."""
        return self.cost("gci0", th.stack([sub, sup], dim=1), True).reshape(-1)
