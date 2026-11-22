"""Objectives: how each axiom form is read as a relation between regions.

An :class:`Objective` needs nothing but a geometry
(:class:`mowl.representation.Geometry`): it asks for the regions of the classes of an
axiom, builds the compound ones (:math:`\\exists r.C`, :math:`C \\sqcap D`) and has the
measure compare them. The default reading is the semantic one:

=============================  ==============================================
:math:`C \\sqsubseteq D`         ``inside(C, D)``
:math:`C \\sqcap D \\sqsubseteq E`  ``inside(C and D, E)``
:math:`C \\sqsubseteq \\exists r.D`  ``inside(C, exists r.D)``
:math:`\\exists r.C \\sqsubseteq D`  ``inside(exists r.C, D)``
:math:`C \\sqsubseteq \\bot`       ``empty(C)``
:math:`C \\sqcap D \\sqsubseteq \\bot`  ``apart(C, D)``
:math:`C(a)`                    ``inside(a, C)``
:math:`r(a, b)`                 ``inside(a, exists r.{b})``
=============================  ==============================================

The two forms about relations are read by the relation model
(:meth:`~mowl.representation.relation_models.Relations.role_inclusion`,
:meth:`~mowl.representation.relation_models.Relations.role_chain`):

=============================  ==============================================
:math:`r \\sqsubseteq s`         ``relations.role_inclusion(r, s, measure, margin)``
:math:`r \\circ t \\sqsubseteq s`  ``relations.role_chain(r, t, s, measure, margin)``
=============================  ==============================================

A negative axiom (``neg=True``) asks for the two regions to be apart, as in DELE. The
options reproduce what publications and their codes do differently; the constructors
:meth:`Objective.elem` and :meth:`Objective.elbe` set them as in those methods, and the
functions that build other published methods (:mod:`mowl.representation.library`) set theirs.
Every option is off by default.
"""

import torch as th

from mowl.representation.regions import ConstructorUnavailable

__all__ = ["Objective", "FORMS", "ROLE_FORMS", "SQUARED_FORMS", "NORM_FORMS"]

#: The concept and assertion forms an objective reads.
FORMS = ("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
         "class_assertion", "object_property_assertion")
#: The forms about relations (EL++): ``r ⊑ s`` (columns ``r, s``) and ``r ∘ t ⊑ s`` (columns
#: ``r, t, s``).
ROLE_FORMS = ("role_inclusion", "role_chain")
#: Forms squared by ``squared=True``.
SQUARED_FORMS = ("gci0", "gci1", "gci3", "gci0_bot", "gci1_bot", "gci3_bot")
#: Forms that carry the unit-norm term of ``unit_norm=True``.
NORM_FORMS = ("gci0", "gci1", "gci2", "gci3", "gci1_bot", "class_assertion",
              "object_property_assertion", "role_inclusion", "role_chain")

# Columns of the classes ("c") and individuals ("i") of a form, for the unit-norm term.
_NORM_COLUMNS = {
    "gci0": (("c", 0), ("c", 1)), "gci1": (("c", 0), ("c", 1), ("c", 2)),
    "gci2": (("c", 0), ("c", 2)), "gci3": (("c", 1), ("c", 2)),
    "gci1_bot": (("c", 0), ("c", 1)),
    # read as the axiom with the bottom class (``bottom="class"``)
    "gci0_bot": (("c", 0), ("c", 1)), "gci3_bot": (("c", 1), ("c", 2)),
    "class_assertion": (("i", 0), ("c", 1)),
    "object_property_assertion": (("i", 0), ("i", 2))}
_ROLE_COLUMNS = {"role_inclusion": (0, 1), "role_chain": (0, 1, 2)}
_NEGATIVES = {"all": frozenset(FORMS), "gci2": frozenset({"gci2", "object_property_assertion"}),
              "gci0": frozenset({"gci0"}), "none": frozenset()}


class Objective:
    """The reading of the axiom forms.

    :param margin: The margin :math:`\\gamma \\ge 0`.
    :param slack: ``False`` (default): the margin is strict, relations have to hold with
        :math:`\\gamma` to spare and a zero loss implies the axiom. ``True``: the margin is
        a slack, as in ELEmbeddings.
    :param epsilon: Smallest extent of a class that must not be empty.
    :param gci3: ``"inclusion"`` (default): :math:`\\exists r.C` lies inside :math:`D`.
        ``"overlap"``: it only has to overlap :math:`D`, as the published losses of
        ELEmbeddings and ELBE have it.
    :param gci2_negative: The negative reading of :math:`C \\sqsubseteq \\exists r.D` (and of
        role assertions). ``"apart"`` (default), ``"overlap"`` (the negative loss of ELBE:
        zero as soon as the two regions overlap by less than the margin), ``"linear"``
        (``apart`` without its hinge, the ELEm baseline of the Box2EL code), ``"delta"``
        (:math:`(\\delta - \\mu)^2` per argument, Box2EL and DELE; the relation model needs
        ``subsumed_negative``), ``"outside"`` (the region sticks out of the other,
        ``measure.outside``), ``"shrunk"`` (``measure.meets_shrunk``, the published loss of
        the prism module) or ``"negative"`` (``measure.negative``).
    :param negatives: Which forms have a negative loss of their own: ``"all"`` (default),
        ``"gci2"`` (:math:`C \\sqsubseteq \\exists r.D` and role assertions: for the others
        ``neg=True`` returns the positive loss, as the original modules do), ``"gci0"``
        (hierarchy methods), ``"none"``, or a collection of form names.
    :param strict_negatives: ``True`` (default): asking for the negative loss of a form that
        has none (``neg=True`` outside ``negatives``) is an error
        (:class:`NotImplementedError`), as in :class:`~mowl.nn.TransBoxModule`: a positive
        loss is not a negative one. ``False``: it gives the positive loss, which is what most
        of the existing modules do (:class:`~mowl.nn.ELModule`: ``neg_capable_gcis`` says
        which forms differ); the functions that rebuild such a module set it.
    :param empty: ``"size"`` (default): the loss of :math:`C \\sqsubseteq \\bot` is the
        size of the region. ``"raw"``: the signed radius parameter, as in the ELEmbeddings
        code (unbounded below).
    :param gci2_offset: Constant added to the positive loss of
        :math:`C \\sqsubseteq \\exists r.D` (``1e-5`` in the ELEmbeddings code).

    Options that change how a form is read (keyword only):

    :param gci0_negative: The negative reading of :math:`C \\sqsubseteq D`: ``"apart"``
        (default), ``"negative"`` (``measure.negative``, the cost of a pair that is not
        ordered: hierarchy methods) or ``"outside"`` (non-inclusion, RegD).
    :param gci2_reading: ``"exists"`` (default): :math:`C` is compared with the region of
        :math:`\\exists r.D` (through the relation model's own ``subsumed`` where it has
        one: head and tail boxes have a second condition). ``"image"``: the image of
        :math:`C` is compared with :math:`D` (the same for bijections; BoxEL differs). Role
        assertions are read in the same way.
    :param bottom: ``"empty"`` (default): :math:`C \\sqsubseteq \\bot` is the region of
        :math:`C` being empty. ``"class"``: the bottom concept is a class of its own that
        the axiom is included in (last column of the data; TransBox, BoxEL).
    :param gci3_bot: ``"exists"`` (default): :math:`\\exists r.C \\sqsubseteq \\bot` is the region
        of :math:`\\exists r.C` being empty. ``"filler"``: the region of :math:`C` is empty
        (the same for a translation). ``gci3_bot_negative`` is the reading of its negative
        (default: the same).
    :param gci1_bot: ``None`` (default): as ``bottom`` says. ``"apart"``: the two regions are
        disjoint. ``"product"``: the volume product :math:`\\mathrm{Vol}(C \\cap D) /
        (\\mathrm{Vol}(C)\\mathrm{Vol}(D))` of the BoxEL code (``measure.product_overlap``).
    :param disjoint_margin: Margin of :math:`C \\sqcap D \\sqsubseteq \\bot` when it differs
        from the margin (``None``: the same).
    :param empty_margin: Pass the margin to ``measure.empty``.
    :param neg_margins: Margin of the negative reading per form, where it differs.
    :param negative_margin: ``"same"`` (default): a negative reading uses the margin in the
        same sense as the positive one (a slack where ``slack=True``). ``"strict"``: always
        as a strict margin, which is what the published losses of ELEmbeddings do.
    :param squared: Square the positive loss of :data:`SQUARED_FORMS` (``True``), or of
        the forms given (Box2EL).
    :param delta: The :math:`\\delta` of ``gci2_negative="delta"``.
    :param weights: A factor per form (Concept2Box weights its two views).
    :param ranking: ``{form: gamma}``: the negative cost of that form is the positive cost
        of the corrupted axiom minus ``gamma``, so that the margin ranking loss
        :math:`[\\gamma + f(\\delta) - f(\\delta')]_+` of EIKE is ``relu(positive - negative)``;
        :class:`~mowl.models.ComposedEL` uses that loss for these forms.
    :param unit_norm: Add :math:`|\\lVert x \\rVert - 1|` of every class (individual) centre
        of an axiom to its loss, and of the shift vectors of a role axiom (EmEL++ Eq. 1-10,
        TransBox's code). ``True``: the forms of :data:`NORM_FORMS`; or a collection of forms.
    :param tolerance_weight: Weight of the term :math:`+\\sigma_r` added to a positive loss
        with an existential restriction when the relation model has a tolerance
        (EmEL(var): Mohapatra et al. 2021, Eq. 7-8). Negative readings use the preimage
        without tolerance.
    :param gci0_shift: A tensor, or a function returning one (or ``None``), added to the
        centre of :math:`C` in :math:`C \\sqsubseteq D` (the learned "SubClassOf" vector of
        the EmEL++ Keras code).
    :param freeze_size_negatives: Compute the negative cost of :math:`C \\sqsubseteq D` with the
        sizes of the regions detached (box lattice model: :math:`\\Delta` is not updated
        through negative pairs); the class model must support ``detach_size``.
    :param inside_margin: The margin of ``inside`` when it is not the margin of the negative
        readings (hierarchy methods: no margin on stated pairs); ``None``: as ``margin``
        and ``slack`` say.
    """

    def __init__(self, margin=0.0, slack=False, epsilon=0.01, gci3="inclusion",
                 gci2_negative="apart", negatives="all", empty="size", gci2_offset=0.0, *,
                 gci0_negative="apart", gci2_reading="exists", bottom="empty",
                 gci3_bot="exists", gci3_bot_negative=None, gci1_bot=None,
                 disjoint_margin=None, empty_margin=False, neg_margins=None,
                 negative_margin="same", squared=False, delta=2.0, weights=None,
                 ranking=None, unit_norm=False, tolerance_weight=1.0, gci0_shift=None,
                 freeze_size_negatives=False, inside_margin=None, strict_negatives=True):
        _check(gci3, ("inclusion", "overlap"), "gci3")
        _check(gci2_negative, ("apart", "overlap", "linear", "delta", "outside", "shrunk",
                               "negative"), "gci2_negative")
        _check(empty, ("size", "raw"), "empty")
        _check(gci0_negative, ("apart", "negative", "outside"), "gci0_negative")
        _check(gci2_reading, ("exists", "image"), "gci2_reading")
        _check(bottom, ("empty", "class"), "bottom")
        _check(gci3_bot, ("exists", "filler"), "gci3_bot")
        _check(gci3_bot_negative, (None, "exists", "filler"), "gci3_bot_negative")
        _check(gci1_bot, (None, "apart", "product"), "gci1_bot")
        _check(negative_margin, ("same", "strict"), "negative_margin")
        if isinstance(negatives, str):
            if negatives not in _NEGATIVES:
                raise ValueError(f"negatives must be one of {sorted(_NEGATIVES)} or a "
                                 f"collection of forms.")
            self.neg_forms = _NEGATIVES[negatives]
            self.negatives = negatives
        else:
            self.neg_forms = frozenset(negatives)
            self.negatives = "forms"
        self.margin = float(margin)
        self.slack = bool(slack)
        self.epsilon = float(epsilon)
        self.gci3_reading = gci3
        self.gci2_negative = gci2_negative
        self.empty = empty
        self.gci2_offset = float(gci2_offset)
        self.gci0_negative = gci0_negative
        self.gci2_reading = gci2_reading
        self.bottom = bottom
        self.gci3_bot_reading = gci3_bot
        self.gci3_bot_negative = gci3_bot_negative or gci3_bot
        self.gci1_bot_reading = gci1_bot
        self.disjoint_margin = None if disjoint_margin is None else float(disjoint_margin)
        self.empty_margin = bool(empty_margin)
        self.neg_margins = dict(neg_margins or {})
        self.negative_margin = negative_margin
        self.squared = SQUARED_FORMS if squared is True else tuple(squared or ())
        self.delta = float(delta)
        self.weights = dict(weights or {})
        self.ranking = None if ranking is None else dict(ranking)
        self.unit_norm = NORM_FORMS if unit_norm is True else tuple(unit_norm or ())
        self.tolerance_weight = float(tolerance_weight)
        self.gci0_shift = gci0_shift
        self.freeze_size_negatives = bool(freeze_size_negatives)
        self.inside_margin = None if inside_margin is None else float(inside_margin)
        self.strict_negatives = bool(strict_negatives)

    # -- the readings of publications -----------------------------------------------------
    @classmethod
    def elem(cls, margin=0.1, all_negatives=False, linear_negative=False, epsilon=0.01,
             gci3="overlap", **options):
        """ELEmbeddings (Kulmanov et al. 2019) as implemented in mOWL."""
        options.setdefault("negative_margin", "strict")
        options.setdefault("strict_negatives", False)
        return cls(margin, slack=True, epsilon=epsilon, gci3=gci3,
                   gci2_negative="linear" if linear_negative else "apart",
                   negatives="all" if all_negatives else "gci2", empty="raw",
                   gci2_offset=10e-6, **options)

    @classmethod
    def elbe(cls, margin=0.1, all_negatives=False, epsilon=0.01, gci3="overlap", **options):
        """ELBE (Peng et al. 2022) as implemented in mOWL."""
        options.setdefault("strict_negatives", False)
        return cls(margin, slack=False, epsilon=epsilon, gci3=gci3,
                   gci2_negative="overlap", negatives="all" if all_negatives else "gci2",
                   **options)

    @classmethod
    def hierarchy(cls, margin=0.0, inside_margin=0.0, freeze_size_negatives=False, **options):
        """The reading of the methods that embed a hierarchy: only :math:`C \\sqsubseteq D`
        between named classes has a negative loss, the cost of a pair that is *not* ordered
        (``measure.negative``), with ``margin`` (:math:`\\alpha` of order embeddings,
        :math:`\\gamma` of the cones, :math:`\\mu` of disks; measures that have none ignore
        it); stated pairs have no margin (``inside_margin``, as in all published losses of
        these methods).

        :param freeze_size_negatives: Compute the cost of a negative pair with the sizes of
            the regions detached (the box lattice model: :math:`\\Delta` is not updated
            through negative pairs).
        """
        options.setdefault("negatives", "gci0")
        return cls(margin, inside_margin=inside_margin, gci0_negative="negative",
                   freeze_size_negatives=freeze_size_negatives, **options)

    # -- margins ----------------------------------------------------------------------------
    @property
    def m(self):
        """Margin of ``inside`` and ``meets``: positive when strict, negative as a slack."""
        if self.inside_margin is not None:
            return self.inside_margin
        return -self.margin if self.slack else self.margin

    @property
    def nm(self):
        """Margin of the negative readings (``apart``, ``outside``, ...), in the sense of
        :attr:`negative_margin`."""
        return self.m if self.negative_margin == "same" and self.inside_margin is None \
            else self.margin

    def _nm(self, form):
        """Margin of the ``apart`` reading of the negative of ``form``."""
        return self.neg_margins.get(form, self.nm)

    def _sm(self, form):
        """Margin of the other negative readings of ``form`` (overlap, outside, ...), always
        in the strict sense."""
        return self.neg_margins.get(form, self.margin)

    def _m_of(self, g):
        """The margin of ``inside`` in the geometry ``g`` (a hook: it may depend on the mode
        of the geometry)."""
        return self.m

    def _neg(self, neg, form=None):
        """Whether ``neg=True`` has a negative loss of its own for ``form`` (``None``: for
        every form)."""
        if form is None:
            return bool(neg) and self.neg_forms == _NEGATIVES["all"]
        return bool(neg) and form in self.neg_forms

    # -- the forms --------------------------------------------------------------------------
    def cost(self, g, form, rows, neg=False):
        """The loss of the axioms ``rows`` of form ``form`` in the geometry ``g``,
        ``(B, 1)``."""
        if form not in FORMS and form not in ROLE_FORMS:
            raise ValueError(f"Unknown axiom form {form!r}; known: {FORMS + ROLE_FORMS}.")
        if self.ranking is not None and neg and form in self.ranking:
            return self._form_cost(g, form, rows, False) - self.ranking[form]
        if neg and self.strict_negatives and form not in self.neg_forms:
            raise NotImplementedError(
                f"There is no negative loss of {form!r} (only {sorted(self.neg_forms)}); "
                f"use strict_negatives=False to get the positive loss instead.")
        cost = self._form_cost(g, form, rows, neg)
        return cost * self.weights[form] if form in self.weights else cost

    def _form_cost(self, g, form, rows, neg):
        cost = getattr(self, form)(g, rows, neg)
        if form in self.squared and not self._neg(neg, form):
            cost = cost.square()
        return self._with_norms(cost, g, form, rows)

    def _norms(self, g, form, rows):
        """The unit-norm term of ``form`` (``0.0`` if the form has none)."""
        if form not in self.unit_norm:
            return 0.0
        if form in _ROLE_COLUMNS:
            vectors = g.relation_model.vectors
            return sum(th.abs(th.linalg.norm(vectors(rows[:, c]), dim=1, keepdim=True) - 1)
                       for c in _ROLE_COLUMNS[form])
        regions = [(g.classes if kind == "c" else g.individuals)(rows[:, c])
                   for kind, c in _NORM_COLUMNS[form]]
        return sum(th.abs(th.linalg.norm(r.centre, dim=1, keepdim=True) - 1)
                   for r in regions)

    def _with_norms(self, cost, g, form, rows):
        return cost + self._norms(g, form, rows) if form in self.unit_norm else cost

    def _classes(self, g, ids, frozen=False):
        return g.classes(ids, True) if frozen else g.classes(ids)

    def _not_included(self, g, c, d, margin):
        reading = self.gci0_negative
        if reading == "negative":
            return g.measure.negative(c, d, self._sm("gci0"))
        if reading == "outside":
            return g.measure.outside(c, d, self._sm("gci0"))
        return g.measure.apart(c, d, margin)

    def gci0(self, g, rows, neg=False):
        frozen = self.freeze_size_negatives and neg and "gci0" in self.neg_forms
        c, d = self._classes(g, rows[:, 0], frozen), self._classes(g, rows[:, 1], frozen)
        if self._neg(neg, "gci0"):
            cost = self._not_included(g, c, d, self._nm("gci0"))
        else:
            shift = self.gci0_shift() if callable(self.gci0_shift) else self.gci0_shift
            cost = g.measure.inside(c if shift is None else c.translate(shift), d,
                                    self._m_of(g))
        return cost

    def gci1(self, g, rows, neg=False):
        c, d, e = (g.classes(rows[:, i]) for i in range(3))
        if self._neg(neg, "gci1"):
            cost = g.measure.intersection_apart(c, d, e, self._nm("gci1"))
        else:
            cost = g.measure.intersection_inside(c, d, e, self._m_of(g))
        return cost

    def _sides(self, g, sub, relation, filler, sharp):
        """The two regions an existential axiom compares: the region of the subclass and
        that of :math:`\\exists r.D`, or (``gci2_reading="image"``) the image of the
        subclass and the filler."""
        if self.gci2_reading == "image":
            return g.relation_model.image(relation, sub), filler
        rel = g.relation_model
        if sharp and rel is not None:
            return sub, rel.sharp_preimage(relation, filler)
        return sub, g.exists(relation, filler)

    def _tolerance(self, g, relation):
        rel = g.relation_model
        if rel is None or not self.tolerance_weight:
            return 0.0
        tolerance = rel.tolerance(relation)
        return self.tolerance_weight * tolerance if not isinstance(tolerance, float) else 0.0

    def _existential(self, g, sub, relation, filler, neg, form="gci2"):
        """``sub`` against :math:`\\exists r.\\text{filler}`."""
        rel = g.relation_model
        if not neg:
            m = self._m_of(g)
            if self.gci2_reading == "exists" and rel is not None \
                    and type(rel).subsumed is not _base_subsumed():
                cost = rel.subsumed(g.measure, sub, relation, filler, m)
            else:
                left, right = self._sides(g, sub, relation, filler, False)
                cost = g.measure.inside(left, right, m)
            return cost + self.gci2_offset + self._tolerance(g, relation)
        reading = self.gci2_negative
        if reading == "delta":
            return rel.subsumed_negative(g.measure, sub, relation, filler, self.margin,
                                         self.delta)
        left, right = self._sides(g, sub, relation, filler, True)
        measure = g.measure
        if reading == "overlap":
            return measure.meets_all(left, right, -self._sm(form))
        if reading == "linear":
            return _unhinged_apart(left, right, self._nm(form))
        if reading == "outside":
            return measure.outside(left, right, self._sm(form))
        if reading == "shrunk":
            return measure.meets_shrunk(left, right, -self._sm(form))
        if reading == "negative":
            return measure.negative(left, right, self._sm(form))
        return measure.apart(left, right, self._nm(form))

    def gci2(self, g, rows, neg=False):
        sub, filler = g.classes(rows[:, 0]), g.classes(rows[:, 2])
        cost = self._existential(g, sub, rows[:, 1], filler, self._neg(neg, "gci2"))
        return cost

    def gci3(self, g, rows, neg=False):
        neg = self._neg(neg, "gci3")
        filler, d = g.classes(rows[:, 1]), g.classes(rows[:, 2])
        exists = g.relation_model.sharp_preimage(rows[:, 0], filler) \
            if neg and g.relation_model is not None else g.exists(rows[:, 0], filler)
        if neg and self.gci2_negative == "delta":
            cost = (self.delta - g.measure.meets(exists, d, self.margin)).square()
        elif neg:
            cost = g.measure.apart(exists, d, self._nm("gci3"))
        else:
            if self.gci3_reading == "overlap":
                cost = g.measure.meets(exists, d, self._m_of(g))
            else:
                cost = g.measure.inside(exists, d, self._m_of(g))
            cost = cost + self._tolerance(g, rows[:, 0])
        return cost

    def _bottom(self, g, c, neg, form="gci0_bot"):
        if self._neg(neg, form):
            return g.measure.nonempty(c, self.epsilon)
        if self.empty == "raw":
            if getattr(c, "raw", None) is None:
                raise TypeError("empty='raw' needs a class model that keeps a signed size.")
            return c.raw
        if self.empty_margin:
            return g.measure.empty(c, self.margin)
        return g.measure.empty(c)

    def gci0_bot(self, g, rows, neg=False):
        if self.bottom == "class":
            return self.gci0(g, rows[:, :2], neg)
        return self._bottom(g, g.classes(rows[:, 0]), neg)

    def gci3_bot(self, g, rows, neg=False):
        if self.bottom == "class":
            return self.gci3(g, rows[:, :3], neg)
        reading = self.gci3_bot_negative if self._neg(neg, "gci3_bot") else self.gci3_bot_reading
        if reading == "filler":
            return self._bottom(g, g.classes(rows[:, 1]), neg, "gci3_bot")
        # the region of exists r.C is empty
        return self._bottom(g, g.exists(rows[:, 0], g.classes(rows[:, 1])), neg, "gci3_bot")

    def gci1_bot(self, g, rows, neg=False):
        if self.bottom == "class" and self.gci1_bot_reading is None:
            return self.gci1(g, rows[:, :3], neg)
        c, d = g.classes(rows[:, 0]), g.classes(rows[:, 1])
        if self._neg(neg, "gci1_bot"):
            cost = g.measure.intersection_nonempty(c, d, -self.margin, self.epsilon)
        elif self.gci1_bot_reading == "product":
            cost = g.measure.product_overlap(c, d)
        else:
            margin = self.disjoint_margin if self.disjoint_margin is not None else self.nm
            cost = g.measure.apart(c, d, margin)
        return cost

    def class_assertion(self, g, rows, neg=False):
        a, c = g.individuals(rows[:, 0]), g.classes(rows[:, 1])
        if self._neg(neg, "class_assertion"):
            cost = g.measure.apart(a, c, self._nm("class_assertion"))
        else:
            cost = g.measure.member(a, c, self._m_of(g))
        return cost

    def object_property_assertion(self, g, rows, neg=False):
        a, b = g.individuals(rows[:, 0]), g.individuals(rows[:, 2])
        cost = self._existential(g, a, rows[:, 1], b,
                                 self._neg(neg, "object_property_assertion"),
                                 "object_property_assertion")
        return cost

    # -- role axioms ------------------------------------------------------------------------
    def _relations_with(self, g):
        relations = g.relation_model
        if relations is None or not relations.reads_role_axioms:
            raise ConstructorUnavailable(
                f"{type(relations).__name__} does not define role_inclusion(...) and "
                f"role_chain(...), so the role axioms cannot be read.")
        return relations

    def role_inclusion(self, g, rows, neg=False):
        """:math:`r \\sqsubseteq s`. There is no negative reading: ``neg=True`` returns the
        positive loss."""
        relations = self._relations_with(g)
        cost = relations.role_inclusion(rows[:, 0], rows[:, 1], g.measure,
                                        self._m_of(g)).reshape(-1, 1)
        return cost

    def role_chain(self, g, rows, neg=False):
        """:math:`r \\circ t \\sqsubseteq s`."""
        relations = self._relations_with(g)
        cost = relations.role_chain(rows[:, 0], rows[:, 1], rows[:, 2], g.measure,
                                    self._m_of(g)).reshape(-1, 1)
        return cost


def _check(value, allowed, name):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {[a for a in allowed if a is not None]}"
                         f"{' or None' if None in allowed else ''}, not {value!r}.")


def _base_subsumed():
    from mowl.representation.relation_models import Relations
    return Relations.subsumed


def _unhinged_apart(a, b, margin):
    """``apart`` without the hinge, for ball regions (the only published use)."""
    distance = th.linalg.norm(a.centre - b.centre, dim=1, keepdim=True)
    return a.radius + b.radius - distance + margin
