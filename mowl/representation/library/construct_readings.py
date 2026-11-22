"""Readings of constructed models (:class:`~mowl.representation.construction.Reading`).

==============  ====================  =================================================
key             reads                 regions
==============  ====================  =================================================
element sets    finite / temporal     :class:`ElementSets` (the model itself)
half-axes       finite / temporal     :class:`AxisCones` (al-cones, JAIR Defs. 2, 7)
intervals       finite / temporal     :class:`IntervalUnions` (TGDK 2024, Def. 12)
simplex         finite / temporal     :class:`SimplexFaces` (KR 2018, Prop. 3)
boxes           a box model           :class:`~mowl.representation.CornerBox`
==============  ====================  =================================================

A reading of a finite model places *cells* in space (an element, or a class of elements
with the same memberships), reads every class name as the region of the cells of its
extension, every role as the relation of the cells of its pairs (:class:`CellRelations`)
and every individual as the point of its cells. Every one of them satisfies a normal-form
axiom iff the finite model does (see :mod:`mowl.representation.construction`); they
differ in what else they can express: complement and union (element sets, cones,
intervals: yes; faces of a simplex: no) and points of several cells (element sets, cones
in a doubled space, faces: yes; intervals: no).

A finite model cannot be read as single boxes in general: boxes have the Helly property
(boxes that meet pairwise have a common point), finite models do not, so a reading of
cells as boxes would make ``A ⊓ B ⊓ C ⊑ ⊥`` false where the model makes it true. The box
reading reads the box model of a solver (``box milp``) instead, and refuses finite models
with that reason; the cell readings refuse box models (they have no cells).
"""

import torch as th

from mowl.representation.construction import (
    BOTTOM, CellRelations, FiniteModel, Guarantee, Points, ReadModel, Reading,
    TemporalModel, register_reading)
from mowl.representation.library.construct_regions import (
    AxisCones, ElementSets, IntervalUnions, SimplexFaces, simplex)
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["CellReading", "ElementSetReading", "HalfAxisReading", "IntervalReading",
           "SimplexReading", "BoxReading"]

_PRESERVES = Guarantee(
    "soundness", "A normal-form axiom (also with ⊥, role inclusions and chains) holds in "
    "the regions iff it holds in the finite model (φ(η) = M).", "EL++ normal forms")
_BOOLEAN = Guarantee(
    "closure", "Closed under intersection, union and complement: every Boolean concept "
    "(and ∃r.C) is read exactly as in the finite model.", "ALC")


class CellReading(Reading):
    """Base class of the readings of finite (and temporal) models as regions of cells.

    :param merge: Place elements with the same memberships (at every time point) in one
        cell. Only for models without roles (elements with the same memberships may have
        different successors). The temporal al-cone construction does this (one half-axis
        per distinct trace).
    """

    accepts = (FiniteModel, TemporalModel)
    region = None

    def __init__(self, merge=False):
        self.merge = bool(merge)

    def why_not(self, produced):
        return (" A box model is a set of boxes of R^d: it has no domain elements to place "
                "as cells.")

    # -- cells -----------------------------------------------------------------------------
    @staticmethod
    def _steps(built):
        return built.steps if isinstance(built, TemporalModel) else (built,)

    def cells(self, built):
        """``(number of cells, cell of every element)``."""
        size = built.size
        if not self.merge:
            return size, list(range(size))
        steps = self._steps(built)
        if any(s.roles and any(s.roles.values()) for s in steps):
            raise ConstructorUnavailable("Elements can only be merged in a model without "
                                         "roles (merge=True).")
        names = sorted({n for s in steps for n in s.concepts}, key=str)
        signature = [tuple(frozenset(n for n in names if x in s.concepts[n]) for s in steps)
                     for x in range(size)]
        first = {}
        for x, sig in enumerate(signature):
            first.setdefault(sig, len(first))
        return len(first), [first[sig] for sig in signature]

    def _mask(self, elements, cell, n):
        out = th.zeros(n, dtype=th.bool)
        out[[cell[x] for x in elements]] = True
        return out

    def masks(self, built, names, source, cell, n):
        """``(N, T, n)``: the cells of every name at every time point."""
        steps = self._steps(built)
        out = th.zeros(len(names), len(steps), n, dtype=th.bool)
        for i, name in enumerate(names):
            key = source.name_of(name)
            for t, step in enumerate(steps):
                out[i, t] = self._mask(step.extension(key), cell, n)
        return out

    def pairs(self, built, roles, cell, n):
        steps = self._steps(built)
        if len(steps) > 1:
            if roles:
                raise ConstructorUnavailable("Roles of temporal models are not read.")
            return None
        if not roles:
            return None
        out = th.zeros(len(roles), n, n, dtype=th.bool)
        for i, r in enumerate(roles):
            for x, y in built.role(r):
                out[i, cell[x], cell[y]] = True
        return out

    # -- regions ---------------------------------------------------------------------------
    def geometry(self, built, n):
        """The geometry of the family (keyword arguments of the region class)."""
        return {}

    def read(self, built, names, roles, individuals, source):
        n, cell = self.cells(built)
        geometry = self.geometry(built, n)
        successor = built.successors() if isinstance(built, TemporalModel) \
            else th.zeros(1, dtype=th.long)
        T = len(self._steps(built))

        def region(mask):
            return self.region(mask, successor=successor, **geometry)

        classes = region(self.masks(built, names, source, cell, n))
        universe = region(th.ones(1, T, n, dtype=th.bool))
        empty = region(th.zeros(1, T, n, dtype=th.bool))
        pairs = self.pairs(built, roles, cell, n)
        relations = None
        if pairs is not None:
            relations = CellRelations(pairs, self.relation_part).bind_template(empty)
        points = None
        if individuals:
            model = built.steps[0] if isinstance(built, TemporalModel) else built
            masks, centres = [], []
            for a in individuals:
                if a not in model.individuals:
                    # an individual the construction does not know: in no region
                    masks.append(th.zeros(n, dtype=th.bool))
                    centres.append(None)
                    continue
                cells = {cell[x] for x in model.element_set(a)}
                masks.append(self._mask([x for x in range(model.size)
                                         if cell[x] in cells], cell, n))
                centres.append(universe.point(cells))
            known = [c for c in centres if c is not None]
            if not known:
                known = [universe.point({0})] if self.region is not IntervalUnions \
                    else [th.zeros(1, dtype=th.float64)]
            centres = [th.full_like(known[0], float("nan")) if c is None else c
                       for c in centres]
            mask = th.stack(masks).unsqueeze(1).expand(-1, T, -1)
            points = Points(th.stack(centres), mask)
        return ReadModel(classes, relations, points, universe, empty, list(names),
                         list(roles), list(individuals))


@register_reading
class ElementSetReading(CellReading):
    """The finite model itself: a class is the set of its elements."""

    key = "element sets"
    region = ElementSets
    class_part = "fixed by reasoning"
    relation_part = "fixed by reasoning"
    individual_part = "canonical element"
    constructors = {"conjunction": "exact", "existential": "exact", "negation": "exact",
                    "disjunction": "exact"}
    guarantees = (_PRESERVES, _BOOLEAN)


@register_reading
class HalfAxisReading(CellReading):
    """Cells on the half-axes of :math:`\\mathbb{R}^D`, :math:`D = \\lceil n / 2 \\rceil`
    (cell ``k`` on half-axis ``k``; for an odd number of cells the last half-axis carries a
    copy of cell 0): a class is the al-cone spanned by the half-axes of its cells, a role
    the incidence matrix of its pairs (JAIR Def. 7), an individual the unit vector of its
    half-axis (the construction "finite model to al-cones" of
    :mod:`mowl.geometric.cones`). An individual of several cells (Boolean atoms with
    ``faithful=True``) makes the space doubled (JAIR Prop. 8)."""

    key = "half-axes"
    region = AxisCones
    class_part = "axis-aligned cone"
    relation_part = "matrix"
    individual_part = "region"
    constructors = {"conjunction": "exact", "existential": "exact", "negation": "exact",
                    "disjunction": "exact"}
    guarantees = (_PRESERVES, _BOOLEAN, Guarantee(
        "soundness", "halfAxes(C) = {h : f(h) ∈ C^I} for every ALC concept C, so the cones "
        "satisfy exactly the GCIs and assertions of the finite model.", "ALC"))

    def __init__(self, merge=False, double=None):
        super().__init__(merge)
        self.double = double

    @property
    def accepts(self):
        from mowl.geometric.cones import ALConeInterpretation
        return (FiniteModel, TemporalModel, ALConeInterpretation)

    def geometry(self, built, n):
        dim = max(1, (n + 1) // 2)
        halfaxis = list(range(n)) + [0] * (2 * dim - n)
        model = built.steps[0] if isinstance(built, TemporalModel) else built
        double = model.partial_individuals if self.double is None else self.double
        if not double:
            if model.partial_individuals:
                raise ConstructorUnavailable(
                    "An individual of several cells needs the doubled space (double=True, "
                    "JAIR Prop. 8).")
            return {"halfaxis_cell": th.tensor(halfaxis)}
        if any(model.roles.values()):
            raise ConstructorUnavailable("The doubled space of individuals of several cells "
                                         "is read without roles (JAIR Prop. 8).")
        doubled = [halfaxis[2 * (h // 4) + h % 2] for h in range(4 * dim)]
        return {"halfaxis_cell": th.tensor(doubled), "doubled": True,
                "original_cell": th.tensor(halfaxis)}

    def read(self, built, names, roles, individuals, source):
        if isinstance(built, (FiniteModel, TemporalModel)):
            return super().read(built, names, roles, individuals, source)
        return self._read_cones(built, names, roles, individuals)

    def _read_cones(self, interp, names, roles, individuals):
        """An al-cone interpretation given by its cones: one cell per half-axis."""
        import numpy as np
        dim = interp.dim
        geometry = {"halfaxis_cell": th.arange(2 * dim)}
        zero = th.zeros(1, dtype=th.long)

        def region(mask):
            return AxisCones(mask, successor=zero, **geometry)

        bits = [interp.concepts[n].bits if n in interp.concepts else np.zeros(2 * dim, bool)
                for n in names]
        classes = region(th.as_tensor(np.stack(bits)).reshape(len(names), 1, -1)
                         if names else th.zeros(0, 1, 2 * dim, dtype=th.bool))
        universe = region(th.ones(1, 1, 2 * dim, dtype=th.bool))
        empty = region(th.zeros(1, 1, 2 * dim, dtype=th.bool))
        relations = None
        if roles:
            pairs = th.stack([th.as_tensor(interp.roles[r]).T for r in roles])
            relations = CellRelations(pairs, self.relation_part).bind_template(empty)
        points = None
        if individuals:
            centre = th.as_tensor(np.stack([interp.individuals[a] for a in individuals]),
                                  dtype=th.float64)
            support = th.zeros(len(individuals), 2 * dim, dtype=th.bool)
            support[:, 0::2] = centre > 1e-12
            support[:, 1::2] = centre < -1e-12
            points = Points(centre, support.unsqueeze(1))
        return ReadModel(classes, relations, points, universe, empty, list(names),
                         list(roles), list(individuals))


@register_reading
class IntervalReading(CellReading):
    """Cell ``k`` is the interval :math:`[k + 1 - w, k + 1 + w]` of the line (``w`` =
    ``half_width`` < 1/2): a class is the union of the intervals of its cells, a role the
    union of the products of the intervals of its pairs, an individual the centre of its
    interval (Lacerda, Ozaki & Guimarães, TGDK 2024, Def. 12; the 1-D case of the
    multiboxes of Federico, Leemhuis & Kutz, JOWO 2026, Thm. 3)."""

    key = "intervals"
    region = IntervalUnions
    class_part = "union of intervals"
    relation_part = "region over pairs"
    individual_part = "point"
    constructors = {"conjunction": "exact", "existential": "exact", "negation": "exact",
                    "disjunction": "exact"}
    guarantees = (_PRESERVES, _BOOLEAN)

    def __init__(self, half_width=0.25, merge=False):
        super().__init__(merge)
        if not 0 < half_width < 0.5:
            raise ValueError("half_width must lie in (0, 1/2): the intervals are disjoint.")
        self.half_width = float(half_width)

    def geometry(self, built, n):
        centre = th.arange(1, n + 1, dtype=th.float64)
        return {"lower": centre - self.half_width, "upper": centre + self.half_width}


@register_reading
class SimplexReading(CellReading):
    """Cell ``k`` is vertex ``k`` of the standard simplex of :math:`\\mathbb{R}^{n - 1}`: a
    class is the face spanned by the vertices of its cells, a role the convex hull of the
    concatenated vertex pairs, an individual the vertex of its cell (the centroid of its
    cells); Gutiérrez-Basulto & Schockaert, KR 2018, Prop. 3, in simplex coordinates.

    :param thicken: Thickened faces (:class:`~mowl.geometric.regions.ThickFace` with
        :math:`\\varepsilon = 1 / 2n`): full-dimensional, for propositional ontologies (no
        :math:`\\exists r.C`).
    """

    key = "simplex"
    region = SimplexFaces
    class_part = "convex region"
    relation_part = "region over pairs"
    individual_part = "point"
    constructors = {"conjunction": "exact", "existential": "exact", "negation": "none",
                    "disjunction": "none"}
    guarantees = (_PRESERVES, Guarantee(
        "closure", "Faces of a simplex are closed under intersection, not under union or "
        "complement (convex regions).", "EL++"))

    def __init__(self, thicken=False, merge=False):
        super().__init__(merge)
        self.thicken = bool(thicken)

    def geometry(self, built, n):
        return {"points": simplex(n), "eps": 1.0 / (2 * n) if self.thicken else 0.0}


@register_reading
class BoxReading(Reading):
    """The boxes of a box model (:class:`~mowl.geometric.boxes.BoxModel`) as
    :class:`~mowl.representation.CornerBox` regions with their corners, ``TOP`` as the
    universe :math:`[0, M]^d`, ``BOTTOM`` and unknown names as the empty box (corners
    :math:`+\\infty, -\\infty`). No relations (the box models are propositional), no
    individuals."""

    key = "boxes"
    class_part = "box"
    relation_part = "none"
    individual_part = "none"
    constructors = {"conjunction": "exact", "existential": "none", "negation": "none",
                    "disjunction": "none"}
    guarantees = (Guarantee("closure", "Boxes are closed under intersection, not under "
                            "union or complement.", "propositional EL⊥"),)

    @property
    def region(self):
        from mowl.representation.regions import CornerBox
        return CornerBox

    @property
    def accepts(self):
        from mowl.geometric.boxes import BoxModel
        return (BoxModel,)

    def why_not(self, produced):
        return (" A finite model cannot in general be read as single boxes: boxes have the "
                "Helly property (pairwise meeting boxes have a common point) and finite "
                "models do not; the 'box milp' source searches a box model with a solver.")

    def read(self, built, names, roles, individuals, source):
        import numpy as np
        from mowl.representation.construction import _corner_box
        if roles:
            raise ConstructorUnavailable("Box models have no roles.")

        def boxes(regions):
            low = th.as_tensor(np.stack([r.lo for r in regions]), dtype=th.float64)
            high = th.as_tensor(np.stack([r.hi for r in regions]), dtype=th.float64)
            return _corner_box(low, high)

        regions = [built.class_region(source.name_of(n)) for n in names]
        universe = boxes([built.universe])
        empty = boxes([built.class_region(BOTTOM)])
        classes = boxes(regions) if regions else universe
        return ReadModel(classes, None, None, universe, empty, list(names), [], [])
