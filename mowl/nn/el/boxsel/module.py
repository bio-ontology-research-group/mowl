"""BoxSEL module: the BoxEL geometry plus the statistical conditionals of Zhu et al. (2024)."""
import torch as th

from mowl.nn.el.boxel.module import BoxELModule


class ConditionalExpressions:
    """The part of a geometry that reads *conditionals* :math:`(D \\mid C)[l, u]` (BoxSEL). The
    class expressions are nested tuples without names: ``("atom",)`` is a class,
    ``("and", e1, e2, ...)`` a conjunction and ``("some", e)`` an existential restriction,
    with the indices of the classes and relations of the expression, left to right, as the
    ``(B, k)`` tensor ``leaves``
    (:func:`mowl.nn.el.boxsel.losses.compile_expression` makes both from names). It is a
    mixin of :class:`~mowl.nn.BoxSELModule` and of the geometry of the method ``boxsel``
    (:class:`~mowl.representation.library.boxrel_methods.ConditionalGeometry`); the
    objective of the geometry reads the form ``"conditional"``."""

    def expression(self, struct, leaves, pos=0):
        """``(region, next position)`` of the expression ``struct`` for a batch."""
        kind = struct[0]
        if kind == "atom":
            return self.classes(leaves[:, pos]), pos + 1
        if kind == "some":
            inner, nxt = self.expression(struct[1], leaves, pos + 1)
            return self.exists(leaves[:, pos], inner), nxt
        region = None
        for sub in struct[1:]:
            part, pos = self.expression(sub, leaves, pos)
            region = part if region is None else region.intersect(part)[0]
        return region, pos

    def conditional_loss(self, conclusion, premise, leaves_d, leaves_c, lower, upper):
        """Loss of a batch of conditionals ``(D | C)[l, u]`` whose conclusions and premises
        have the structures ``conclusion`` and ``premise``; ``leaves_*`` are index tensors.
        ``(B, 1)``: :math:`l\\,\\mathrm{Vol}(C) \\le \\mathrm{Vol}(D \\cap C) \\le
        u\\,\\mathrm{Vol}(C)` as the measure reads it."""
        return self.cost("conditional", (conclusion, premise, leaves_d, leaves_c, lower,
                                         upper))

    @th.no_grad()
    def probability(self, conclusion, premise, leaves_d, leaves_c):
        """Proportion of the premise region that lies in the conclusion region, exact."""
        c, _ = self.expression(premise, leaves_c)
        d, _ = self.expression(conclusion, leaves_d)
        return self.measure.probability(c, d)


class BoxSELModule(ConditionalExpressions, BoxELModule):
    """BoxEL boxes with the SEL losses on top.

    The deterministic normal forms (``gci0`` ... ``gci3``, bottom and assertion forms) are
    the BoxEL losses, which BoxSEL takes over unchanged (Appendix D of the paper). On top,
    :meth:`conditional_loss` encodes a probabilistic conditional :math:`(D \\mid C)[l, u]` as
    :math:`l\\,\\mathrm{Vol}(C) \\le \\mathrm{Vol}(D \\cap C) \\le u\\,\\mathrm{Vol}(C)`, and
    :meth:`regularization_loss` is :math:`L_{loc}` (side length bound ``beta``) plus,
    optionally, :math:`L_{vol}`. The parts are those of the builder ``boxsel`` of
    :mod:`mowl.representation.library.boxrel_methods` (those of BoxEL, the objective with the
    conditional, the two penalties); ``module.parts`` says which.

    :param beta: Side length :math:`\\beta` of the cube the boxes should lie in.
    :param location_weight: Weight of :math:`L_{loc}`.
    :param volume_weight: Weight of :math:`L_{vol}` (as printed in the paper).
    """

    def __init__(self, *args, beta=1.0, location_weight=1.0, volume_weight=0.0, **kwargs):
        from mowl.representation.library.boxrel_methods import boxsel
        super().__init__(*args, _builder=boxsel, beta=beta, location_weight=location_weight,
                         volume_weight=volume_weight, **kwargs)
        self._beta, self._location_weight, self._volume_weight = (
            beta, location_weight, volume_weight)

    def _penalties(self):
        """The two penalties of the geometry follow the three numbers (a weight of 0 leaves
        the penalty out)."""
        from mowl.representation.library.boxrel_regularizers import (
            LocationPenalty, VolumePenalty)
        penalties = []
        if self._location_weight:
            penalties.append(LocationPenalty(self._beta, weight=self._location_weight))
        if self._volume_weight:
            penalties.append(VolumePenalty(self._beta, weight=self._volume_weight))
        self.regularizers = tuple(penalties)

    @property
    def beta(self):
        """Side :math:`\\beta` of the cube the boxes should lie in."""
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
        self._penalties()

    @property
    def location_weight(self):
        return self._location_weight

    @location_weight.setter
    def location_weight(self, value):
        self._location_weight = value
        self._penalties()

    @property
    def volume_weight(self):
        return self._volume_weight

    @volume_weight.setter
    def volume_weight(self, value):
        self._volume_weight = value
        self._penalties()


__all__ = ["BoxSELModule", "ConditionalExpressions"]
