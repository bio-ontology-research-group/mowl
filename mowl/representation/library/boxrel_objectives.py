"""Objectives and the geometry class of the boxes group.

:class:`RoleObjective` extends the core :class:`~mowl.representation.Objective` by the two
role axiom forms and by readings that boxes with affine relations need; the objectives of
the four methods set their published options on top of it. :class:`RoleGeometry` is the
:class:`~mowl.representation.ComposedGeometry` that routes the role forms to the objective.
"""

import torch as th

from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.boxrel_boxes import signed_intersection
from mowl.representation.objectives import FORMS, Objective
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["ROLE_FORMS", "RoleObjective", "RoleGeometry", "TransBoxObjective",
           "BoxELObjective", "BoxSELObjective", "ELKEmObjective"]

#: The role axiom forms: ``r ⊑ s`` (columns ``r, s``) and ``r ∘ t ⊑ s`` (columns
#: ``r, t, s``).
ROLE_FORMS = ("role_inclusion", "role_chain")


class RoleObjective(Objective):
    """The core objective, plus

    * the role forms, which the relation model answers
      (``role_inclusion(sub, sup, measure, margin)`` and
      ``role_chain(first, second, sup, measure, margin)``); a negative role axiom is
      not defined and returns the positive loss;
    * ``bottom``: ``"empty"`` (default) reads :math:`C \\sqsubseteq \\bot` as the region of
      :math:`C` being empty; ``"class"`` reads the bottom concept as a class of its own that
      the axiom is included in: the last column of the data is the class (TransBox and
      BoxEL do this);
    * ``gci2_reading``: ``"exists"`` (default) compares :math:`C` with the region of
      :math:`\\exists r.D`; ``"image"`` compares the image of :math:`C` under the relation
      with :math:`D` (the same for relations that are bijections; BoxEL and its volume ratio
      differ), and reads role assertions in the same way;
    * ``assertion_negatives``: ``False`` makes ``neg=True`` return the positive loss of a
      role assertion, as the modules of TransBox do.
    """

    def __init__(self, *args, bottom="empty", gci2_reading="exists",
                 assertion_negatives=True, **kwargs):
        super().__init__(*args, **kwargs)
        if bottom not in ("empty", "class"):
            raise ValueError("bottom must be 'empty' or 'class'.")
        if gci2_reading not in ("exists", "image"):
            raise ValueError("gci2_reading must be 'exists' or 'image'.")
        self.bottom = bottom
        self.gci2_reading = gci2_reading
        self.assertion_negatives = bool(assertion_negatives)

    @property
    def neg_capable(self):
        """The forms that have a negative version of their own (``neg=True``)."""
        if self.negatives == "all":
            return frozenset(FORMS)
        return frozenset({"gci2", "object_property_assertion"} if self.assertion_negatives
                         else {"gci2"})

    def cost(self, g, form, rows, neg=False):
        if form in ROLE_FORMS:
            return getattr(self, form)(g, rows, neg)
        return super().cost(g, form, rows, neg)

    # -- role forms -------------------------------------------------------------------------
    @staticmethod
    def _relations_with(g, method):
        relations = g.relation_model
        if relations is None or not hasattr(relations, method):
            raise ConstructorUnavailable(
                f"{type(relations).__name__} does not define {method}(...), so the "
                f"role axiom '{method}' cannot be read.")
        return relations

    def role_inclusion(self, g, rows, neg=False):
        self._relations_with(g, "role_inclusion")
        return g.relation_model.role_inclusion(rows[:, 0], rows[:, 1], g.measure,
                                               self._m_of(g)).reshape(-1, 1)

    def role_chain(self, g, rows, neg=False):
        self._relations_with(g, "role_chain")
        return g.relation_model.role_chain(rows[:, 0], rows[:, 1], rows[:, 2], g.measure,
                                           self._m_of(g)).reshape(-1, 1)

    def _m_of(self, g):
        return self.m

    # -- the forms with a bottom concept -----------------------------------------------------
    def gci0_bot(self, g, rows, neg=False):
        if self.bottom == "class":
            return Objective.gci0(self, g, rows[:, :2], neg)
        return super().gci0_bot(g, rows, neg)

    def gci1_bot(self, g, rows, neg=False):
        if self.bottom == "class":
            return Objective.gci1(self, g, rows[:, :3], neg)
        return super().gci1_bot(g, rows, neg)

    def gci3_bot(self, g, rows, neg=False):
        if self.bottom == "class":
            return Objective.gci3(self, g, rows[:, :3], neg)
        return super().gci3_bot(g, rows, neg)

    # -- existential restrictions ------------------------------------------------------------
    def _existential(self, g, sub, relation, filler, neg):
        if self.gci2_reading == "exists":
            return super()._existential(g, sub, relation, filler, neg)
        image = g.relation_model.image(relation, sub)
        if not neg:
            return g.measure.inside(image, filler, self.m) + self.gci2_offset
        if self.gci2_negative == "overlap":
            return g.measure.meets(image, filler, -self.margin)
        return g.measure.apart(image, filler, self.margin)

    def object_property_assertion(self, g, rows, neg=False):
        a, b = g.individuals(rows[:, 0]), g.individuals(rows[:, 2])
        neg = neg and self.assertion_negatives
        if self.gci2_reading == "image":
            return self._existential(g, a, rows[:, 1], b, neg)
        return super().object_property_assertion(g, rows, neg)


class RoleGeometry(ComposedGeometry):
    """A :class:`~mowl.representation.ComposedGeometry` that also reads the role axioms
    (``role_inclusion``, ``role_chain``) with its objective, which must be a
    :class:`RoleObjective`."""

    role_axiom_capable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.objective, "neg_capable"):
            self.neg_capable_gcis = self.objective.neg_capable

    def get_loss_function(self, gci_name):
        if gci_name in ROLE_FORMS:
            return lambda rows, neg=False: self.cost(gci_name, rows, neg)
        return super().get_loss_function(gci_name)


def _role_loss(form):
    def loss(self, rows, neg=False):
        return self.cost(form, rows, neg)
    loss.__name__ = f"{form}_loss"
    loss.__doc__ = f"Loss of the ``{form}`` axioms ``rows`` (see the objective)."
    return loss


for _form in ROLE_FORMS:
    setattr(RoleGeometry, f"{_form}_loss", _role_loss(_form))
del _form


# ==========================================================================================
class TransBoxObjective(RoleObjective):
    """TransBox (Yang et al. 2025) as implemented in mOWL.

    :param formulation: ``"mowl"``: the losses of mOWL. ``"reference"``: the losses of the
        authors' code, which add :math:`|\\lVert c\\rVert - 1|` of every class centre of
        an axiom to its loss, read disjointness as separation in every coordinate, give
        the negative of :math:`C \\sqsubseteq \\exists r.D` as the overlap with the
        unenhanced box, and leave the half-width of :math:`\\exists r^{all}.D` unclamped.
        (:math:`C \\sqcap D \\sqsubseteq E` as an overlap is an option of the measure,
        :class:`MaskedBounds`.)
    :param enhanced: Train :math:`C \\sqsubseteq \\exists r.D` against the semantic
        enhancement :math:`\\exists r^{all}.D`.
    """

    def __init__(self, margin=0.1, enhanced=True, formulation="mowl"):
        if formulation not in ("mowl", "reference"):
            raise ValueError("formulation must be 'mowl' or 'reference'.")
        super().__init__(margin, slack=False, negatives="gci2", bottom="class",
                         assertion_negatives=False)
        self.enhanced = bool(enhanced)
        self.formulation = formulation

    @property
    def reference(self):
        return self.formulation == "reference"

    @staticmethod
    def _centre_norm(*regions):
        total = 0
        for r in regions:
            total = total + (th.linalg.norm(r.centre, dim=1, keepdim=True) - 1).abs()
        return total

    def gci0(self, g, rows, neg=False):
        cost = Objective.gci0(self, g, rows, False)
        if self.reference:
            cost = cost + self._centre_norm(g.classes(rows[:, 0]), g.classes(rows[:, 1]))
        return cost

    def gci1(self, g, rows, neg=False):
        cost = Objective.gci1(self, g, rows, False)
        if self.reference:
            cost = cost + self._centre_norm(*(g.classes(rows[:, i]) for i in range(3)))
        return cost

    def gci1_bot(self, g, rows, neg=False):
        if not self.reference:
            return self.gci1(g, rows[:, :3])
        c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
        return g.measure.apart(c, d, self.margin) + self._centre_norm(c, d)

    def gci3(self, g, rows, neg=False):
        cost = Objective.gci3(self, g, rows, False)
        if self.reference:
            cost = cost + self._centre_norm(g.classes(rows[:, 1]), g.classes(rows[:, 2]))
        return cost

    def gci0_bot(self, g, rows, neg=False):
        return Objective.gci0(self, g, rows[:, :2], False)

    def gci3_bot(self, g, rows, neg=False):
        return Objective.gci3(self, g, rows[:, :3], False)

    def gci2(self, g, rows, neg=False):
        sub, filler = g.classes(rows[:, 0]), g.classes(rows[:, 2])
        relations = g.relation_model
        exists = relations.preimage(rows[:, 1], filler)
        if neg:
            if self.reference:
                cost = g.measure.apart(sub, exists, -self.margin)
            else:
                target = self._target(g, rows[:, 1], filler, exists)
                cost = (1 - g.measure.apart(sub, target, -self.margin)) ** 2
        else:
            cost = g.measure.inside(sub, self._target(g, rows[:, 1], filler, exists), self.m)
        if self.reference:
            cost = cost + self._centre_norm(sub, filler)
        return cost

    def _target(self, g, relation, filler, exists):
        if not self.enhanced:
            return exists
        return g.relation_model.preimage_all(relation, filler, clamp=not self.reference)


# ==========================================================================================
class BoxELObjective(RoleObjective):
    """BoxEL (Xiong et al. 2022) as implemented in mOWL: no margin; :math:`C \\sqsubseteq
    \\exists r.D` and role assertions compare the image of :math:`C` with :math:`D`; a
    bottom concept is a class; only :math:`C \\sqsubseteq \\exists r.D` and role assertions
    have a negative loss (the share of the image inside :math:`D`, which a volume ratio
    gives as ``apart``).

    :param gci1_bot: ``"volume product"``: :math:`\\mathrm{Vol}(C \\cap D) /
        (\\mathrm{Vol}(C)\\,\\mathrm{Vol}(D))` as the BoxEL code has it; ``"apart"``: the
        share of :math:`C` inside :math:`D` (the measure's reading of disjointness).
    """

    def __init__(self, margin=0.0, gci1_bot="volume product", **kwargs):
        if gci1_bot not in ("volume product", "apart"):
            raise ValueError("gci1_bot must be 'volume product' or 'apart'.")
        kwargs.setdefault("gci3", "inclusion")
        kwargs.setdefault("negatives", "gci2")
        kwargs.setdefault("bottom", "class")
        kwargs.setdefault("gci2_reading", "image")
        super().__init__(margin, **kwargs)
        self.gci1_bot_reading = gci1_bot

    def gci1_bot(self, g, rows, neg=False):
        if self.gci1_bot_reading == "volume product":
            return g.measure.product_overlap(g.classes(rows[:, 0]), g.classes(rows[:, 1]))
        return g.measure.apart(g.classes(rows[:, 0]), g.classes(rows[:, 1]), self.margin)


class BoxSELObjective(BoxELObjective):
    """BoxEL's objective and the statement of BoxSEL: a *conditional*
    :math:`(D \\mid C)[l, u]`, "between :math:`l` and :math:`u` of :math:`C` lies in
    :math:`D`", read as :math:`l\\,\\mathrm{Vol}(C) \\le \\mathrm{Vol}(C \\cap D) \\le
    u\\,\\mathrm{Vol}(C)`. The statement is not an axiom of OWL; it is given as the two
    class expressions (:meth:`RoleGeometry.expression`) and the two bounds."""

    def cost(self, g, form, rows, neg=False):
        if form == "conditional":
            return self.conditional(g, *rows)
        return super().cost(g, form, rows, neg)

    def conditional(self, g, conclusion, premise, leaves_d, leaves_c, lower, upper):
        c, _ = g.expression(premise, leaves_c)
        d, _ = g.expression(conclusion, leaves_d)
        return g.measure.conditional(c, d, lower, upper)


# ==========================================================================================
class ELKEmObjective(RoleObjective):
    """ELK-Em (Singh et al. 2026), Table 2 of the paper.

    :param margin: :math:`\\gamma`.
    :param abox_threshold: Target score :math:`\\tau` of a corrupted assertion: its loss is
        pushed up to :math:`-\\log\\tau`.
    :param gci2_overlap: Add :math:`\\ell_{\\cap+}(R_r, B_D)` to the loss of
        :math:`C \\sqsubseteq \\exists r.D` (the authors' code does; the paper does not).
    :param evaluation_margin: Margin used when the geometry is in evaluation mode
        (``module.eval()``); ``None``: the same as in training. ELK-Em uses 0.
    """

    def __init__(self, margin=0.1, abox_threshold=0.4, gci2_overlap=False,
                 evaluation_margin=0.0):
        super().__init__(margin, slack=False, negatives="gci2")
        self.abox_threshold = float(abox_threshold)
        self.gci2_overlap = bool(gci2_overlap)
        self.evaluation_margin = evaluation_margin

    @property
    def neg_capable(self):
        return frozenset({"class_assertion", "object_property_assertion"})

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
        cost = g.measure.inside(signed_intersection(a, b), e, m)
        over = g.measure.meets(a, b, m)
        trivial = g.measure.is_empty(a) | g.measure.is_empty(b)
        return cost + th.where(trivial, th.zeros_like(over), over)

    def gci1_bot(self, g, rows, neg=False):
        return g.measure.apart(g.classes(rows[:, 0]), g.classes(rows[:, 1]), self._m_of(g))

    def gci2(self, g, rows, neg=False):
        m = self._m_of(g)
        filler = g.classes(rows[:, 2])
        cost = g.measure.inside(g.classes(rows[:, 0]), g.exists(rows[:, 1], filler), m)
        if self.gci2_overlap:
            cost = cost + g.measure.meets(g.relation_model.range(rows[:, 1]), filler, m)
        return cost

    def gci3(self, g, rows, neg=False):
        m = self._m_of(g)
        filler = g.classes(rows[:, 1])
        cost = g.measure.inside(g.exists(rows[:, 0], filler), g.classes(rows[:, 2]), m)
        over = g.measure.meets(g.relation_model.range(rows[:, 0]), filler, m)
        return cost + th.where(g.measure.is_empty(filler), th.zeros_like(over), over)

    def gci3_bot(self, g, rows, neg=False):
        return g.measure.apart(g.relation_model.range(rows[:, 0]), g.classes(rows[:, 1]),
                               self._m_of(g))

    def class_assertion(self, g, rows, neg=False):
        cost = g.measure.inside(g.individuals(rows[:, 0]), g.classes(rows[:, 1]),
                                self._m_of(g))
        return self._hinge(cost) if neg else cost

    def object_property_assertion(self, g, rows, neg=False):
        m = self._m_of(g)
        a, b = g.individuals(rows[:, 0]), g.individuals(rows[:, 2])
        relations = g.relation_model
        cost = g.measure.inside(a, relations.image_with_error(rows[:, 1], b), m)
        cost = cost + g.measure.inside(b, relations.range(rows[:, 1]), m)
        return self._hinge(cost) if neg else cost
