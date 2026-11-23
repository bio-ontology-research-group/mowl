"""The objectives of the boxes group that are readings of their own: TransBox, BoxSEL and ELK-Em.

Everything the four methods share is an option of the core
:class:`~mowl.representation.Objective` (``bottom="class"``, ``gci2_reading="image"``,
``unit_norm``, ``gci1_bot``, ``negatives``) or a method of the relation model (the role axioms);
:func:`boxel_objective` sets the published options of BoxEL. What remains:

* :class:`TransBoxObjective`: the semantic enhancement :math:`\\exists r^{all}.D` and the two
  readings of the negative of :math:`C \\sqsubseteq \\exists r.D`.
* :class:`BoxSELObjective`: the statement type of a conditional probability, which is not an
  axiom of OWL.
* :class:`ELKEmObjective`: every form is read with the tuples of ELK-Em (an evaluation margin,
  empty boxes inside everything, hinged negative assertions).
"""

from mowl.representation.objectives import Objective
import torch as th

__all__ = ["boxel_objective", "TransBoxObjective", "BoxSELObjective", "ELKEmObjective"]


def boxel_objective(margin=0.0, gci1_bot="product", **options):
    """BoxEL (Xiong et al. 2022) as implemented in mOWL: no margin; :math:`C \\sqsubseteq
    \\exists r.D` and role assertions compare the image of :math:`C` with :math:`D`; a bottom
    concept is a class; only :math:`C \\sqsubseteq \\exists r.D` and role assertions have a
    negative loss (the share of the image inside :math:`D`, which a volume ratio gives as
    ``apart``).

    :param gci1_bot: ``"product"``: :math:`\\mathrm{Vol}(C \\cap D) /
        (\\mathrm{Vol}(C)\\,\\mathrm{Vol}(D))` as the BoxEL code has it; ``"apart"``: the
        share of :math:`C` inside :math:`D` (the measure's reading of disjointness).
    """
    options.setdefault("gci3", "inclusion")
    options.setdefault("negatives", "gci2")
    options.setdefault("bottom", "class")
    options.setdefault("gci2_reading", "image")
    options.setdefault("strict_negatives", False)
    return Objective(margin, gci1_bot=gci1_bot, **options)


# ==========================================================================================
class TransBoxObjective(Objective):
    """TransBox (Yang et al. 2025) as implemented in mOWL: the core objective with
    ``bottom="class"`` and negatives for :math:`C \\sqsubseteq \\exists r.D` only, and its own
    reading of that axiom.

    :param formulation: ``"mowl"``: the losses of mOWL. ``"reference"``: the losses of the
        authors' code, which add :math:`|\\lVert c\\rVert - 1|` of every class centre of
        an axiom to its loss (``unit_norm``), read disjointness as separation in every
        coordinate, give the negative of :math:`C \\sqsubseteq \\exists r.D` as the overlap
        with the unenhanced box, and leave the half-width of :math:`\\exists r^{all}.D`
        unclamped. (:math:`C \\sqcap D \\sqsubseteq E` as an overlap is an option of the
        measure, :class:`MaskedBounds`.)
    :param enhanced: Train :math:`C \\sqsubseteq \\exists r.D` against the semantic
        enhancement :math:`\\exists r^{all}.D`.
    :param reference_bottom_losses: With ``formulation="reference"``: ``"code"`` (default)
        reads :math:`C \\sqsubseteq \\bot` and :math:`\\exists r.C \\sqsubseteq \\bot` as
        the code reads :math:`C \\sqsubseteq D` with the bottom class as :math:`D`, including
        the centre-norm term; ``"mowl"``: without that term, as mOWL did before.
    :param strict_negatives: A negative loss of a form that has none is an error (default,
        as in :class:`~mowl.nn.TransBoxModule`); ``False`` gives the positive loss.
    """

    def __init__(self, margin=0.1, enhanced=True, formulation="mowl",
                 reference_bottom_losses="code", strict_negatives=True):
        if reference_bottom_losses not in ("code", "mowl"):
            raise ValueError("reference_bottom_losses must be 'code' or 'mowl'.")
        if formulation not in ("mowl", "reference"):
            raise ValueError("formulation must be 'mowl' or 'reference'.")
        reference = formulation == "reference"
        super().__init__(margin, slack=False, negatives=("gci2",), bottom="class",
                         gci1_bot="apart" if reference else None,
                         strict_negatives=strict_negatives,
                         unit_norm=("gci0", "gci1", "gci2", "gci3", "gci1_bot")
                         + (("gci0_bot", "gci3_bot") if reference_bottom_losses == "code"
                            else ()) if reference else False)
        self.enhanced = bool(enhanced)
        self.formulation = formulation
        self.reference_bottom_losses = reference_bottom_losses

    @property
    def reference(self):
        return self.formulation == "reference"

    def gci2(self, g, rows, neg=False):
        sub, filler = g.classes(rows[:, 0]), g.classes(rows[:, 2])
        exists = g.exists(rows[:, 1], filler)
        if neg:
            if self.reference:
                return g.measure.apart(sub, exists, -self.margin)
            target = self._target(g, rows[:, 1], filler, exists)
            return (1 - g.measure.apart(sub, target, -self.margin)) ** 2
        return g.measure.inside(sub, self._target(g, rows[:, 1], filler, exists), self.m)

    def _target(self, g, relation, filler, exists):
        if not self.enhanced:
            return exists
        return g.relation_model.operation("preimage_all", "enhanced=True")(
            relation, filler, clamp=not self.reference)


# ==========================================================================================
class BoxSELObjective(Objective):
    """BoxEL's objective and the statement of BoxSEL: a *conditional*
    :math:`(D \\mid C)[l, u]`, "between :math:`l` and :math:`u` of :math:`C` lies in
    :math:`D`", read as :math:`l\\,\\mathrm{Vol}(C) \\le \\mathrm{Vol}(C \\cap D) \\le
    u\\,\\mathrm{Vol}(C)`. The statement is not an axiom of OWL; it is given as the two
    class expressions (:meth:`ConditionalGeometry.expression`) and the two bounds. All other
    forms are read as BoxEL does (:func:`boxel_objective`)."""

    def __init__(self, margin=0.0, gci1_bot="product", **options):
        options.setdefault("gci3", "inclusion")
        options.setdefault("negatives", "gci2")
        options.setdefault("bottom", "class")
        options.setdefault("gci2_reading", "image")
        options.setdefault("strict_negatives", False)
        super().__init__(margin, gci1_bot=gci1_bot, **options)

    def cost(self, g, form, rows, neg=False):
        if form == "conditional":
            return self.conditional(g, *rows)
        return super().cost(g, form, rows, neg)

    def conditional(self, g, conclusion, premise, leaves_d, leaves_c, lower, upper):
        c, _ = g.expression(premise, leaves_c)
        d, _ = g.expression(conclusion, leaves_d)
        return g.measure.conditional(c, d, lower, upper)


# ==========================================================================================
class ELKEmObjective(Objective):
    """ELK-Em (Singh et al. 2026), Table 2 of the paper.

    :param margin: :math:`\\gamma`.
    :param abox_threshold: Target score :math:`\\tau` of a corrupted assertion: its loss is
        pushed up to :math:`-\\log\\tau`.
    :param gci2_overlap: Add :math:`\\ell_{\\cap+}(R_r, B_D)` to the loss of
        :math:`C \\sqsubseteq \\exists r.D` (the authors' code does; the paper does not).
    :param evaluation_margin: Margin used when the geometry is in evaluation mode
        (``module.eval()``); ``None``: the same as in training. ELK-Em uses 0.
    :param trivial_disjointness: A disjointness row with an empty conjunct (a class fixed to
        the empty box) holds trivially and costs 0 (default). ``False``: the cost of the
        disjointness measure, which charges the margin for the empty box and cannot be
        reduced (the authors' code; selected by ``variant="reference"`` of the method).
    """

    def __init__(self, margin=0.1, abox_threshold=0.4, gci2_overlap=False,
                 evaluation_margin=0.0, trivial_disjointness=True):
        super().__init__(margin, slack=False, negatives=("class_assertion",
                                                         "object_property_assertion"),
                         strict_negatives=False)
        self.abox_threshold = float(abox_threshold)
        self.gci2_overlap = bool(gci2_overlap)
        self.trivial_disjointness = bool(trivial_disjointness)
        self.evaluation_margin = evaluation_margin

    def _m_of(self, g):
        if self.evaluation_margin is not None and not g.training:
            return float(self.evaluation_margin)
        return self.margin

    def _hinge(self, loss):
        delta = -th.log(th.tensor(self.abox_threshold, dtype=loss.dtype, device=loss.device))
        return th.relu(delta - loss)

    def gci0(self, g, rows, neg=False):
        return g.measure.inside(g.classes(rows[:, 0]), g.classes(rows[:, 1]), self._m_of(g))

    def gci0_bot(self, g, rows, neg=False):
        return g.measure.empty(g.classes(rows[:, 0]), self._m_of(g))

    def gci1(self, g, rows, neg=False):
        m = self._m_of(g)
        a, b, e = (g.classes(rows[:, i]) for i in range(3))
        region, _ = g.measure.intersect(a, b)    # boxes: the signed intersection
        cost = g.measure.inside(region, e, m)
        over = g.measure.meets(a, b, m)
        trivial = g.measure.is_empty(a) | g.measure.is_empty(b)
        return cost + th.where(trivial, th.zeros_like(over), over)

    def _trivial_to_zero(self, cost, *empty):
        if not self.trivial_disjointness:
            return cost
        trivial = empty[0]
        for e in empty[1:]:
            trivial = trivial | e
        return th.where(trivial, th.zeros_like(cost), cost)

    def gci1_bot(self, g, rows, neg=False):
        a, b = g.classes(rows[:, 0]), g.classes(rows[:, 1])
        cost = g.measure.apart(a, b, self._m_of(g))
        return self._trivial_to_zero(cost, g.measure.is_empty(a), g.measure.is_empty(b))

    @staticmethod
    def _range(g, ids):
        """The range of the relations ``ids`` (``None``: the whole space, for a relation
        model onto the space, which constrains nothing)."""
        return g.relation_model.operation("range", "ELKEmObjective")(ids)

    def gci2(self, g, rows, neg=False):
        m = self._m_of(g)
        filler = g.classes(rows[:, 2])
        cost = g.measure.inside(g.classes(rows[:, 0]), g.exists(rows[:, 1], filler), m)
        if self.gci2_overlap:
            box = self._range(g, rows[:, 1])
            if box is not None:
                cost = cost + g.measure.meets(box, filler, m)
        return cost

    def gci3(self, g, rows, neg=False):
        m = self._m_of(g)
        filler = g.classes(rows[:, 1])
        cost = g.measure.inside(g.exists(rows[:, 0], filler), g.classes(rows[:, 2]), m)
        box = self._range(g, rows[:, 0])
        if box is None:              # the whole space meets every non-empty filler
            return cost
        over = g.measure.meets(box, filler, m)
        return cost + th.where(g.measure.is_empty(filler), th.zeros_like(over), over)

    def gci3_bot(self, g, rows, neg=False):
        filler = g.classes(rows[:, 1])
        box = self._range(g, rows[:, 0])
        if box is None:              # the whole space is apart only from an empty filler
            cost = g.measure.empty(filler, self._m_of(g))
        else:
            cost = g.measure.apart(box, filler, self._m_of(g))
        return self._trivial_to_zero(cost, g.measure.is_empty(filler))

    def class_assertion(self, g, rows, neg=False):
        cost = g.measure.inside(g.individuals(rows[:, 0]), g.classes(rows[:, 1]),
                                self._m_of(g))
        return self._hinge(cost) if neg else cost

    def object_property_assertion(self, g, rows, neg=False):
        m = self._m_of(g)
        a, b = g.individuals(rows[:, 0]), g.individuals(rows[:, 2])
        relations = g.relation_model
        image = relations.operation("image_with_error", "ELKEmObjective")
        cost = g.measure.inside(a, image(rows[:, 1], b), m)
        box = self._range(g, rows[:, 1])
        if box is not None:
            cost = cost + g.measure.inside(b, box, m)
        return self._hinge(cost) if neg else cost
