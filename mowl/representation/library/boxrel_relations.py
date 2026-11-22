"""Relation models of the boxes group: a box of translations (TransBox), an affine map
(BoxEL, BoxSEL) and an affine map with a range and an error box (ELK-Em).

Besides :meth:`image` and :meth:`preimage` they have the two methods the role axioms need,
which a relation model answers because only it knows what a relation is:

* ``role_inclusion(sub, sup, measure, margin)``: cost of :math:`r \\sqsubseteq s`,
* ``role_chain(first, second, sup, measure, margin)``: cost of
  :math:`r_1 \\circ r_2 \\sqsubseteq s`,

with relation indices as arguments and the measure that compares boxes, ``(B, 1)``.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from mowl.representation.class_models import _unit_rows
from mowl.representation.library.boxrel_boxes import _uniform
from mowl.representation.methods import register_relation_model
from mowl.representation.regions import Ball, Box, ConstructorUnavailable
from mowl.representation.relation_models import Relations

__all__ = ["TranslationBoxes", "Affine", "AffineBoxes"]


def _boxes(region, relation):
    if not isinstance(region, Box):
        raise ConstructorUnavailable(
            f"{type(relation).__name__} maps boxes, not {type(region).__name__} regions.")
    return region


def _as_table(parameter):
    """An :class:`~torch.nn.Embedding` whose weight *is* ``parameter`` (a parameter of
    another module is shared, not copied); anything else is returned unchanged."""
    if isinstance(parameter, nn.Parameter):
        table = nn.Embedding(parameter.shape[0], parameter.shape[1])
        table.weight = parameter
        return table
    return parameter


_ACTIVATIONS = {"abs": th.abs, "relu": F.relu, "softplus": F.softplus,
                "none": lambda x: x}


@register_relation_model("box of translations")
class TranslationBoxes(Relations):
    """A relation is a box of shift vectors (TransBox, Yang et al. 2025; Query2Box, Ren et
    al. 2020): the pairs :math:`(x, y)` of the relation are those with :math:`y - x` in the
    box :math:`[c_r - o_r, c_r + o_r]` (``convention="y-x"``, default: a plain translation
    :math:`y = x + c_r` when :math:`o_r = 0`, as :class:`~mowl.representation.Translation`).
    :math:`\\exists r.C` is the box of :math:`C` shifted by :math:`-c_r` and widened by
    :math:`o_r` (the Minkowski sum); the image the box shifted by :math:`c_r` and widened. A
    ball is widened by the norm of :math:`o_r`. A role chain is the sum of the boxes, a role
    inclusion an inclusion of boxes.

    :param vectors: Table of the centres :math:`c_r`; ``offsets``: table of the half-widths
        (stored signed, passed through ``activation``).
    :param activation: ``"abs"`` (default), ``"relu"`` (Query2Box), ``"softplus"`` or
        ``"none"`` (the half-width can then be negative: an option of the published
        Query2Box, not a default).
    :param convention: ``"y-x"`` (default) or ``"x-y"``: the box holds :math:`x - y`, the
        convention of the TransBox code (the sign of :math:`c_r` flips).
    :param inverse_of: A function from relation ids to the ids of the inverse relations
        (rows of the same table): the preimage is then the image under the inverse
        (Query2Box). Without it the inverse is the box shifted the other way.
    :param init_range: ``None`` (default): rows of norm one; a number: centres uniform in
        ``[-r, r]`` and widths in ``[0, r]`` (Query2Box).
    :param clamp_extent: ``True`` (default): the half-width of a box that is shifted and widened
        is cut at zero, so that every region is a box. ``False``: the plain sum, which with
        ``activation="none"`` can be negative (the reference code of Query2Box). The other
        activations never give a negative width, and for them the option changes nothing.
    """

    part = "box of translations"
    accepts = (Box, Ball)

    def __init__(self, size=None, embed_dim=None, vectors=None, offsets=None,
                 activation="abs", convention="y-x", inverse_of=None, init_range=None,
                 clamp_extent=True):
        super().__init__()
        self.clamp_extent = bool(clamp_extent)
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}.")
        if convention not in ("y-x", "x-y"):
            raise ValueError("convention must be 'y-x' or 'x-y'.")
        self.activation_name = activation
        self.activation = _ACTIVATIONS[activation]
        self.sign = 1.0 if convention == "y-x" else -1.0
        self.inverse_of = inverse_of
        vectors, offsets = (_as_table(t) for t in (vectors, offsets))
        if vectors is None:
            vectors = _unit_rows(nn.Embedding(size, embed_dim)) if init_range is None \
                else _uniform(size, embed_dim, -init_range, init_range)
        self.vectors = vectors
        if offsets is None:
            n, d = len(self.vectors.weight), self.vectors.embedding_dim
            offsets = _unit_rows(nn.Embedding(n, d)) if init_range is None \
                else _uniform(n, d, 0, init_range)
        self.offsets = offsets

    def box(self, ids):
        """The box of translations of the relations ``ids``."""
        return Box(self.vectors(ids), self.activation(self.offsets(ids)))

    def related(self, ids, x, y, region=Box):
        """Whether :math:`y - x` (``convention="y-x"``; :math:`x - y` for ``"x-y"``) lies in
        the box of translations, ``(B, P)``. In a geometry of balls (``region=Ball``) the
        translations are the ball of radius :math:`\\lVert o_r \\rVert` around :math:`c_r`
        (the smallest ball around the box), so that the preimage of a ball is the ball
        grown by that radius, as :meth:`preimage` computes it. Without ``inverse_of``."""
        box = self.box(ids)
        shift = self.sign * (y - x)
        if issubclass(region, Ball):
            radius = th.linalg.norm(box.offset, dim=-1, keepdim=True)
            gap = shift.unsqueeze(0) - box.centre.unsqueeze(1)
            return th.linalg.norm(gap, dim=-1) <= radius + 1e-6
        return Box(box.centre, box.offset).contains(shift)

    def _grow(self, region, width):
        if isinstance(region, Box):
            grown = region.grow(width)
            return Box(grown.centre, th.relu(grown.offset)) if self.clamp_extent else grown
        if isinstance(region, Ball):
            return region.grow(th.linalg.norm(width, dim=-1, keepdim=True))
        raise ConstructorUnavailable(
            f"A box of translations maps boxes and balls, not {type(region).__name__}.")

    def image(self, ids, region):
        box = self.box(ids)
        return self._grow(region.translate(self.sign * box.centre), box.offset)

    def preimage(self, ids, region):
        if self.inverse_of is not None:
            return self.image(self.inverse_of(ids), region)
        box = self.box(ids)
        return self._grow(region.translate(-self.sign * box.centre), box.offset)

    def preimage_all(self, ids, region, clamp=True):
        """The box of :math:`\\exists r^{all}.C` of the semantic enhancement of TransBox:
        centre of :math:`C` shifted by :math:`-c_r`, half-width :math:`o_r - o_C`: the box
        of the :math:`C`-boxes that the box of translations reaches all of. ``clamp``: cut
        the half-width at zero (default); the authors' code does not, which gives an
        inverted box where :math:`o_C > o_r`."""
        box = self.box(ids)
        if not isinstance(region, Box):
            raise ConstructorUnavailable("preimage_all needs boxes.")
        offset = box.offset - region.offset
        return Box(region.centre - self.sign * box.centre, th.relu(offset) if clamp else offset)

    def translations(self, ids, measure=None):
        """The set of translations of the relations ``ids`` as a region the measure compares:
        the box, or, for a measure of balls, the ball of radius :math:`\\lVert o_r \\rVert`
        (what the relation is in a geometry of balls, see :meth:`related`)."""
        box = self.box(ids)
        accepts = getattr(measure, "accepts", ())
        if accepts and not issubclass(Box, accepts) and issubclass(Ball, accepts):
            radius = th.linalg.norm(box.offset, dim=-1, keepdim=True)
            return Ball(box.centre, radius)
        return box

    def role_inclusion(self, sub, sup, measure, margin=0.0):
        """The set of translations of ``sub`` lies in that of ``sup``."""
        return measure.inside(self.translations(sub, measure),
                              self.translations(sup, measure), margin)

    def role_chain(self, first, second, sup, measure, margin=0.0):
        """The Minkowski sum of the two sets of translations (the translations of the
        composition) lies in that of ``sup``."""
        a, b = self.translations(first, measure), self.translations(second, measure)
        both = Ball(a.centre + b.centre, a.radius + b.radius) if isinstance(a, Ball) \
            else Box(a.centre + b.centre, a.offset + b.offset)
        return measure.inside(both, self.translations(sup, measure), margin)

    def flat(self):
        """One flat vector per relation, for inspection."""
        return th.cat([self.vectors.weight, self.offsets.weight], -1).detach()


@register_relation_model("affine map")
class Affine(Relations):
    """A relation is a function that scales and shifts every coordinate (BoxEL, Xiong et
    al. 2022): :math:`(x, y) \\in r` iff :math:`y = f_r \\odot x + t_r`. The image of a box is
    a box, :meth:`preimage` (the region of :math:`\\exists r.C`) is the box
    :math:`(C - t_r) / f_r`. A function: every point has one successor.

    :param vectors: Table of the shifts :math:`t_r`; ``scales``: table of the factors
        :math:`f_r` (initialised around 1).
    :param eps: Added to the factors, which keeps the preimage defined.
    :param scale: How the raw factor is read. ``"positive"`` (default, what
        :class:`~mowl.nn.BoxELModule` does with ``positive_scale=True``): :math:`|f| +
        \\epsilon`, a role never inverts a box. ``"mirror"``: the signed factor, a negative one
        mirrors the box (the exact image of the box under the map, which stays a box).
        ``"signed"``: the corners are mapped one by one, so that a negative factor gives an
        inverted box with negative width (the BoxEL code, ``positive_scale=False``). The three
        agree while every factor is positive.
    :param role_chain_composition: ``"in_order"`` (default) composes the two maps
        (:math:`f_2 t_1 + t_2`); ``"as_released"`` is the expression mOWL had before
        (:math:`f_1 t_1 + t_2`).

    Role axioms (the measure and margin are not used): a role is a function, so
    :math:`r \\sqsubseteq s` holds iff the two maps are equal, and
    :math:`r_1 \\circ r_2 \\sqsubseteq s` iff the composed map (first :math:`r_1`, then
    :math:`r_2`) equals the map of :math:`s`. The cost is the distance of the shifts plus a
    term for the factors:

    :param role_scale: ``"equal"`` (default): :math:`\\lVert f_r / f_s - 1 \\rVert`, zero iff
        the factors are equal (the maps are those :meth:`image` uses, with ``eps``).
        ``"at_most"``: :math:`\\lVert [f_r / f_s - 1]_+ \\rVert`, zero as soon as the factor of
        :math:`r` does not exceed that of :math:`s` (the loss of the BoxEL code and of
        :class:`~mowl.nn.BoxELModule`, with the raw factor of :math:`r`); a zero cost then
        does not imply the inclusion (two different functions do not include each other).
    """

    part = "affine map"
    SCALES = ("positive", "mirror", "signed")
    COMPOSITIONS = ("in_order", "as_released")

    functional = True
    accepts = (Box,)

    def __init__(self, size=None, embed_dim=None, vectors=None, scales=None, eps=1e-8,
                 scale="positive", role_chain_composition="in_order", role_scale="equal"):
        super().__init__()
        if role_scale not in ("equal", "at_most"):
            raise ValueError("role_scale must be 'equal' or 'at_most'.")
        self.role_scale = role_scale
        if scale not in self.SCALES:
            raise ValueError(f"scale must be one of {self.SCALES}.")
        if role_chain_composition not in self.COMPOSITIONS:
            raise ValueError(f"role_chain_composition must be one of {self.COMPOSITIONS}.")
        self.vectors = vectors if vectors is not None else _uniform(
            size, embed_dim, -0.1, 0.1)
        self.scales = scales if scales is not None else _uniform(
            len(self.vectors.weight), self.vectors.embedding_dim, 0.9, 1.1)
        self.eps = eps
        self.scale = scale
        self.role_chain_composition = role_chain_composition

    def _raw(self, ids):
        s = self.scales(ids)
        return s.abs() if self.scale == "positive" else s

    def factor(self, ids):
        return self._raw(ids) + self.eps

    def _spread(self, offset, factor):
        return offset * (factor if self.scale == "signed" else factor.abs())

    def successor(self, ids, x):
        """:math:`f_r \\odot x + t_r` of every point ``x`` (``(P, d)``), ``(B, P, d)``."""
        return x.unsqueeze(0) * self.factor(ids).unsqueeze(1) + self.vectors(ids).unsqueeze(1)

    def related(self, ids, x, y, tolerance=1e-5):
        """:math:`y = f_r \\odot x + t_r` (up to ``tolerance``)."""
        return th.linalg.norm(y.unsqueeze(0) - self.successor(ids, x), dim=-1) <= tolerance

    def image(self, ids, region):
        f = self.factor(ids)
        _boxes(region, self)
        return Box(region.centre * f + self.vectors(ids), self._spread(region.offset, f))

    def preimage(self, ids, region):
        f = self.factor(ids)
        _boxes(region, self)
        return Box((region.centre - self.vectors(ids)) / f,
                   self._spread(region.offset, 1 / f))

    def _cost(self, translation_gap, composed_scale, sup_scale):
        excess = composed_scale / sup_scale - 1
        excess = th.relu(excess) if self.role_scale == "at_most" else excess
        return (th.linalg.norm(translation_gap, dim=1, keepdim=True)
                + th.linalg.norm(excess, dim=1, keepdim=True))

    def _role_factor(self, ids):
        """The factor of a role in its axioms: that of the map (``"equal"``), or the raw
        one of the BoxEL code (``"at_most"``)."""
        return self.factor(ids) if self.role_scale == "equal" else self._raw(ids)

    def role_inclusion(self, sub, sup, measure=None, margin=0.0):
        return self._cost(self.vectors(sub) - self.vectors(sup), self._role_factor(sub),
                          self.factor(sup))

    def role_chain(self, first, second, sup, measure=None, margin=0.0):
        f1, f2 = self._role_factor(first), self._role_factor(second)
        scale = f2 if self.role_chain_composition == "in_order" else f1
        shift = scale * self.vectors(first) + self.vectors(second)
        return self._cost(shift - self.vectors(sup), f1 * f2, self.factor(sup))


@register_relation_model("affine map with range and error box")
class AffineBoxes(Relations):
    """A relation is a range box :math:`R_r`, an error box :math:`E_r = [-e_r, e_r]` and a
    diagonal affine map :math:`T_r(x) = m_r x + c_r` with positive slope (ELK-Em, Singh et
    al. 2026): :math:`(x, y) \\in r` iff :math:`y \\in R_r` and :math:`x \\in T_r(y) \\oplus
    E_r`, so that :math:`\\exists r.C` is the box :math:`T_r(C \\cap R_r) \\oplus E_r`
    (:meth:`preimage`).

    :param vectors: Table of the shifts :math:`c_r`. ``range_centres``, ``range_offsets``,
        ``error_offsets`` and ``slopes`` are the other tables; half-widths and slopes are
        used through their absolute value (the slope plus ``eps``).
    :param sound: ``True`` (default): when :math:`C \\cap R_r` is empty the box of
        :math:`\\exists r.C` stays empty. ``False``: the error box is added anyway, which is
        what interval arithmetic does in the authors' code and can make an empty
        restriction non-empty.
    """

    part = "affine map with range and error box"
    accepts = (Box,)

    def __init__(self, size=None, embed_dim=None, vectors=None, range_centres=None,
                 range_offsets=None, error_offsets=None, slopes=None, eps=1e-5, sound=True):
        super().__init__()
        n = len(vectors.weight) if vectors is not None else size
        d = vectors.embedding_dim if vectors is not None else embed_dim
        self.vectors = vectors if vectors is not None else _uniform(n, d, -0.1, 0.1)
        self.range_centres = range_centres if range_centres is not None else _uniform(
            n, d, -0.5, 0.5)
        self.range_offsets = range_offsets if range_offsets is not None else _uniform(
            n, d, -0.1, 0.1)
        self.error_offsets = error_offsets if error_offsets is not None else _uniform(
            n, d, -0.1, 0.1)
        self.slopes = slopes if slopes is not None else _uniform(n, d, 0.9, 1.1)
        self.eps = eps
        self.sound = bool(sound)

    # -- the three parts of a relation ---------------------------------------------------------
    def range(self, ids):
        """The range box :math:`R_r`."""
        return Box(self.range_centres(ids), th.abs(self.range_offsets(ids)))

    def error(self, ids):
        """Half-width :math:`e_r` of the error box."""
        return th.abs(self.error_offsets(ids))

    def slope(self, ids):
        return th.abs(self.slopes(ids)) + self.eps

    def related(self, ids, x, y):
        """:math:`y \\in R_r` and :math:`x \\in T_r(y) \\oplus E_r`, ``(B, P)``."""
        in_range = self.range(ids).contains(y)
        mapped = y.unsqueeze(0) * self.slope(ids).unsqueeze(1) + self.vectors(ids).unsqueeze(1)
        close = (th.abs(x.unsqueeze(0) - mapped) <= self.error(ids).unsqueeze(1) + 1e-6)
        return in_range & close.all(dim=-1)

    def image_with_error(self, ids, region):
        """:math:`T_r(A) \\oplus E_r`, without restricting :math:`A` to the range."""
        return Box(region.centre * self.slope(ids) + self.vectors(ids),
                   region.offset * self.slope(ids) + self.error(ids))

    def preimage(self, ids, region):
        box = self.range(ids)
        _boxes(region, self)
        lower = th.maximum(region.lower, box.lower)
        upper = th.minimum(region.upper, box.upper)
        slope, shift, error = self.slope(ids), self.vectors(ids), self.error(ids)
        offset = (upper - lower) / 2 * slope
        if self.sound:
            empty = (lower > upper).any(dim=1, keepdim=True)
            offset = th.where(empty, offset, offset + error)
        else:
            offset = offset + error
        return Box((lower + upper) / 2 * slope + shift, offset)

    # -- role axioms ---------------------------------------------------------------------------
    def _tuple(self, ids):
        return self.range(ids), self.error(ids), self.slope(ids), self.vectors(ids)

    @staticmethod
    def _tuple_inclusion(r, s, measure, margin):
        (range_r, e_r, m_r, c_r), (range_s, e_s, m_s, c_s) = r, s
        cost = measure.inside(range_r, range_s, margin)
        for y in (range_r.lower, range_r.upper):
            cost = cost + measure.inside(Box(m_r * y + c_r, e_r), Box(m_s * y + c_s, e_s),
                                         margin)
        return cost

    def role_inclusion(self, sub, sup, measure, margin=0.0):
        """:math:`\\ell(R_r, R_s) + \\sum_{y \\in \\{l, u\\}} \\ell(T_r(y) \\oplus E_r,
        T_s(y) \\oplus E_s)` over the end points of the range of :math:`r`."""
        return self._tuple_inclusion(self._tuple(sub), self._tuple(sup), measure, margin)

    def role_chain(self, first, second, sup, measure, margin=0.0):
        """The inclusion of the composition
        :math:`\\langle R_2,\\ m_1 E_2 \\oplus E_1,\\ T_1 \\circ T_2\\rangle` in :math:`s`
        (an over-approximation of the composition, Proposition 4 of the paper)."""
        range_1, e_1, m_1, c_1 = self._tuple(first)
        range_2, e_2, m_2, c_2 = self._tuple(second)
        composed = (range_2, m_1 * e_2 + e_1, m_1 * m_2, m_1 * c_2 + c_1)
        return self._tuple_inclusion(composed, self._tuple(sup), measure, margin)

    def image(self, ids, region):
        raise ConstructorUnavailable("AffineBoxes has no image: the region of "
                                     "exists r.C is the preimage.")
