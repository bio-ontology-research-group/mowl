"""BoxSEL module: the BoxEL geometry plus the statistical conditionals of Zhu et al. (2024)."""
import torch as th

from mowl.nn.el.boxel.module import BoxELModule
from mowl.nn.el.boxsel import losses as S


class BoxSELModule(BoxELModule):
    """BoxEL boxes with the SEL losses on top.

    The deterministic normal forms (``gci0`` ... ``gci3``, bottom and assertion forms) are
    the BoxEL losses, which BoxSEL takes over unchanged (Appendix D of the paper). On top,
    :meth:`conditional_loss` encodes a probabilistic conditional :math:`(D \\mid C)[l, u]` as
    :math:`l\\,\\mathrm{Vol}(C) \\le \\mathrm{Vol}(D \\cap C) \\le u\\,\\mathrm{Vol}(C)`, and
    :meth:`regularization_loss` is :math:`L_{loc}` (side length bound ``beta``) plus,
    optionally, :math:`L_{vol}`.

    :param beta: Side length :math:`\\beta` of the cube the boxes should lie in.
    :param location_weight: Weight of :math:`L_{loc}`.
    :param volume_weight: Weight of :math:`L_{vol}` (as printed in the paper).
    """

    def __init__(self, *args, beta=1.0, location_weight=1.0, volume_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.location_weight = location_weight
        self.volume_weight = volume_weight

    def box(self, struct, leaves):
        """``(m, M)`` of the expression (:mod:`~mowl.nn.el.boxsel.losses`) for a batch."""
        m, big, _ = S.expression_box(struct, leaves, 0, self.min_embedding,
                                     self.delta_embedding, self.relation_embedding,
                                     self.scaling_embedding)
        return m, big

    def conditional_loss(self, conclusion, premise, leaves_d, leaves_c, lower, upper):
        """Loss of a batch of conditionals ``(D | C)[l, u]`` whose conclusions and premises
        have the structures ``conclusion`` and ``premise``; ``leaves_*`` are index tensors."""
        c = self.box(premise, leaves_c)
        d = self.box(conclusion, leaves_d)
        return S.conditional_loss(c, d, lower, upper, self.temperature)

    @th.no_grad()
    def probability(self, conclusion, premise, leaves_d, leaves_c):
        """Proportion of the premise region that lies in the conclusion region, exact."""
        return S.conditional_probability(self.box(premise, leaves_c),
                                         self.box(conclusion, leaves_d))

    def regularization_loss(self):
        out = self.location_weight * S.location_loss(
            self.min_embedding, self.delta_embedding, self.beta)
        if self.volume_weight:
            out = out + self.volume_weight * S.volume_loss(
                self.min_embedding, self.delta_embedding, self.beta)
        return out
