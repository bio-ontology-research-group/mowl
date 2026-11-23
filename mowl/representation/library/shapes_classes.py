"""Class models, individual models and relation models of the shapes group.

* :class:`Points`: individuals as points, as regions of no extent of the family of the
  classes (a ball of radius zero, a box of width zero, ...).
* :class:`OctagonClasses`, :class:`PrismClasses`, :class:`EllipsoidClasses`,
  :class:`SoftBoxClasses`, :class:`ContextualSoftBoxClasses`: new class models.
* :class:`VectorClasses`, :class:`MappedVectors`, :class:`LinkedClasses`: a second space of
  vectors, linked to the first, and the class model that has a region in both.
* :class:`ContextTransform`, :class:`RelationsByRegion`: relations that are a function of
  the class, and relations that are one thing for individuals and another for classes.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from mowl.representation.library import octagon_geometry as G
from mowl.representation.library.shapes_prism_math import supports
from mowl.representation.class_models import Classes
from mowl.representation.methods import register_class_model
from mowl.representation.regions import Ball, Box, ConstructorNotImplemented, ConstructorUnavailable
from mowl.representation.relation_models import Relations
from mowl.representation.library.shapes_regions import (
    Point, OctagonProduct, Prism, Ellipsoid, Vector, Linked, SoftBox)

__all__ = ["Points", "OctagonClasses", "PrismClasses", "EllipsoidClasses", "VectorClasses",
           "MappedVectors", "LinkedClasses", "SoftBoxClasses", "ContextualSoftBoxClasses",
           "ContextTransform", "RelationsByRegion", "table"]


def table(size, dim, init="unit", scale=1.0):
    """A new :class:`torch.nn.Embedding`: ``"unit"`` rows of norm one (default of the
    class models of mOWL), ``"uniform"`` in ``[-scale, scale]``, ``"normal"`` with standard
    deviation ``scale``, or ``"constant"`` ``scale``."""
    emb = nn.Embedding(size, dim)
    with th.no_grad():
        if init == "unit":
            nn.init.uniform_(emb.weight, a=-1, b=1)
            emb.weight /= th.linalg.norm(emb.weight, dim=1, keepdim=True).clamp_min(1e-12)
        elif init == "uniform":
            nn.init.uniform_(emb.weight, a=-scale, b=scale)
        elif init == "normal":
            nn.init.normal_(emb.weight, std=scale)
        elif init == "constant":
            nn.init.constant_(emb.weight, scale)
        else:
            raise ValueError(f"Unknown init {init!r}.")
    return emb


def _zeros(like, width=None):
    shape = like.shape if width is None else (like.shape[0], width)
    return th.zeros(shape, dtype=like.dtype, device=like.device)


# -- individuals ----------------------------------------------------------------------------

#: Region families a point can be made into.
_FAMILIES = {
    "point": (Point, lambda c: Point(c)),
    "ball": (Ball, lambda c: Ball(c, _zeros(c, 1))),
    "box": (Box, lambda c: Box(c, _zeros(c))),
    "octagon": (OctagonProduct, OctagonProduct.of_point),
    "prism": (Prism, Prism.of_point),
    "ellipsoid": (Ellipsoid, lambda c: Ellipsoid(c, th.ones_like(c), _zeros(c, 1))),
    "soft box": (SoftBox, lambda c: SoftBox(c, _zeros(c))),
}


class Points(Classes):
    """Individuals (or classes) as points: a table of vectors, each read as the region of
    no extent of the chosen family, so that the measures of that family compare points with
    classes of the same family.

    :param family: ``"point"`` (the plain :class:`Point` region), ``"ball"``, ``"box"``,
        ``"octagon"``, ``"prism"``, ``"ellipsoid"`` or ``"soft box"``.
    :param centres: An existing table to use instead of a new one.
    :param init, scale: Initialisation of a new table (see :func:`table`).
    :param bump: Whether a point of the family ``"point"`` has a bump vector as well (BoxE,
        a knowledge graph entity); ``bumps`` is an existing table for it.

    Part (vocabulary value): ``point``. Used by: ``boxe``, ``concept2box``, ``elkem``,
    ``expressive``, ``graph_kge``, ``octagon``, ``prism``, ``transbox``.
    """

    part = "point"
    exact_conjunction = False

    def __init__(self, size=None, embed_dim=None, family="point", centres=None, init="unit",
                 scale=1.0, bump=False, bumps=None):
        super().__init__()
        if (bump or bumps is not None) and family != "point":
            raise ValueError("Only points of the family 'point' have a bump.")
        if family not in _FAMILIES:
            raise ValueError(f"Unknown family {family!r}; known: {sorted(_FAMILIES)}.")
        self.family = family
        self.region = _FAMILIES[family][0]
        self._make = _FAMILIES[family][1]
        self.centres = centres if centres is not None else table(size, embed_dim, init, scale)
        if bumps is None and bump:
            bumps = table(len(self), self.embed_dim, init, scale)
        self.bumps = bumps

    def forward(self, ids):
        if self.bumps is not None:
            return Point(self.centres(ids), self.bumps(ids))
        return self._make(self.centres(ids))


# -- octagon products ------------------------------------------------------------------------

class OctagonClasses(Classes):
    r"""A class is a product of 2-D octagons (Charpenay and Schockaert 2024, used here for
    classes): a centre and eight radii per pair of coordinates; the bounds are
    :math:`h_k = n_k \cdot c + |r_k|`, so the centre is always in the region and the class
    is never empty.

    :param embed_dim: Even dimension.
    :param centres, radii: Existing tables (``(n, d)`` and ``(n, 4 d)``).
    :param init_radius: Radii are initialised uniformly in ``[0, init_radius]``.

    Part (vocabulary value): ``octagon product``. Used by: ``octagon``.
    """

    region = OctagonProduct
    part = "octagon product"
    exact_conjunction = True

    def __init__(self, size=None, embed_dim=None, centres=None, radii=None, init_radius=0.5):
        super().__init__()
        if centres is None:
            if embed_dim < 2 or embed_dim % 2:
                raise ValueError(f"embed_dim must be an even int >= 2, got {embed_dim!r}")
            centres = table(size, embed_dim, "uniform")
        self.centres = centres
        if radii is None:
            radii = table(len(self), 4 * self.embed_dim, "uniform", init_radius)
            with th.no_grad():
                radii.weight.abs_()
        self.radii = radii

    def forward(self, ids):
        centre = self.centres(ids)
        pairs = centre.reshape(centre.shape[0], -1, 2)
        radius = th.abs(self.radii(ids)).reshape(centre.shape[0], pairs.shape[1], 8)
        return OctagonProduct(G.from_center_radius(pairs, radius), centre)


# -- prisms ----------------------------------------------------------------------------------

class PrismClasses(Classes):
    r"""A class is a prism: a triangle (centre and inradius) times a box (centre and
    offset). The triangle takes the first two coordinates, the box the remaining
    :math:`d - 2`.

    :param embed_dim: :math:`d \ge 3`.
    :param tri_centres, tri_inradii, box_centres, box_offsets: Existing tables.
    :param resolver: A function from class ids to ``(supports, box centres, box
        half-widths)`` that replaces the tables (the prism module computes the regions of
        defined classes as intersections).

    Part (vocabulary value): ``triangle x box``. Used by: ``prism``,
    ``prism(reading=structural)``.
    """

    region = Prism
    part = "triangle x box"
    exact_conjunction = True

    def __init__(self, size=None, embed_dim=None, tri_centres=None, tri_inradii=None,
                 box_centres=None, box_offsets=None, resolver=None):
        super().__init__()
        if tri_centres is None:
            if embed_dim < 3:
                raise ValueError(f"embed_dim must be an int >= 3, got {embed_dim!r}")
            tri_centres = table(size, 2)
            tri_inradii = table(size, 1, "uniform", 1.0)
            with th.no_grad():
                nn.init.uniform_(tri_inradii.weight, a=0.1, b=1.0)
            box_centres = table(size, embed_dim - 2)
            box_offsets = table(size, embed_dim - 2)
        #: The triangle centres serve as "the" table of the class model.
        self.centres = tri_centres
        self.inradii = tri_inradii
        self.box_centres = box_centres
        self.box_offsets = box_offsets
        self.resolver = resolver

    @property
    def embed_dim(self):
        return 2 + self.box_centres.embedding_dim

    def forward(self, ids):
        if self.resolver is not None:
            h, centre, offset = self.resolver(ids)
            return Prism(h, centre, offset)
        h = supports(self.centres(ids), th.abs(self.inradii(ids)))
        return Prism(h, self.box_centres(ids), th.abs(self.box_offsets(ids)))


# -- ellipsoids and the second space -----------------------------------------------------------

class EllipsoidClasses(Classes):
    """A class is an ellipsoid: a centre, positive axes and a positive scale (EIKE,
    Wang et al. 2024). Axes and scale are stored as the inverse of the softplus.

    :param centres, axes_raw, radius_raw: Existing tables.

    Part (vocabulary value): ``ellipsoid``. Used by: ``eike`` (the first space of its linked
    classes).
    """

    region = Ellipsoid
    part = "ellipsoid"
    exact_conjunction = False
    #: Added to the axes (as in the EIKE code) so that they are never zero.
    epsilon = 1e-6

    def __init__(self, size=None, embed_dim=None, centres=None, axes_raw=None,
                 radius_raw=None):
        super().__init__()
        self.centres = centres if centres is not None else table(
            size, embed_dim, "uniform", 0.5)
        # softplus(0.5413) = 1: unit axes and unit scale.
        self.axes_raw = axes_raw if axes_raw is not None else table(
            len(self), self.embed_dim, "constant", 0.5413)
        self.radius_raw = radius_raw if radius_raw is not None else table(
            len(self), 1, "constant", 0.5413)

    def forward(self, ids):
        return Ellipsoid(self.centres(ids), F.softplus(self.axes_raw(ids)) + self.epsilon,
                         F.softplus(self.radius_raw(ids)))


class VectorClasses(Classes):
    """A class is a vector (what the name means). Needs a measure of vectors
    (:class:`~mowl.representation.library.shapes_measures.CosineOrder`)."""

    region = Vector
    part = "point"
    exact_conjunction = False

    def __init__(self, size=None, embed_dim=None, centres=None):
        super().__init__()
        self.centres = centres if centres is not None else table(
            size, embed_dim, "uniform", 0.5)

    def forward(self, ids):
        return Vector(self.centres(ids))


class MappedVectors(Classes):
    """The vector of an individual in the second space: its point, mapped by a matrix
    (the virtual instance embedding :math:`M i^e` of EIKE).

    :param points: The table of the points (``nn.Embedding``).
    :param matrix: ``(d, d)``: a fixed tensor (identity), or a ``Parameter`` to learn.
    :param owner: ``(module, name)`` of a module that holds the matrix (a parameter or a
        buffer): it is then read from the module each time it is used, so that it follows the
        module when this is moved to another device or cast to another dtype (a buffer is
        replaced then, and a reference taken earlier would keep the old tensor). ``matrix`` is
        not used in that case.
    """

    region = Vector
    part = "point"
    exact_conjunction = False

    def __init__(self, points, matrix=None, owner=None):
        super().__init__()
        self.centres = points
        # a plain tuple in a list: the owner is not registered as a child of this module
        self._owner = None if owner is None else [owner]
        if owner is not None:
            pass
        elif isinstance(matrix, nn.Parameter):
            self.matrix = matrix
        else:
            self.register_buffer("matrix", matrix)

    def _matrix(self, like):
        if self._owner is None:
            return self.matrix
        module, name = self._owner[0]
        return getattr(module, name).to(device=like.device, dtype=like.dtype)

    def forward(self, ids):
        e = self.centres(ids)
        return Vector(e @ self._matrix(e).T)


class LinkedClasses(Classes):
    """Two class models over two spaces, linked by the name: a name has a region in each.
    The first is what it contains, the second what it means. The regions are
    :class:`~mowl.representation.library.shapes_regions.Linked` and are compared by
    :class:`~mowl.representation.library.shapes_measures.LinkedMeasure`.

    :param first, second: The class models (or individual models) of the two spaces.
    """

    region = Linked
    exact_conjunction = False

    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    @property
    def part(self):
        return self.first.part

    @property
    def centres(self):
        return self.first.centres

    def forward(self, ids):
        return Linked(self.first(ids), self.second(ids))


# -- soft boxes --------------------------------------------------------------------------------

class SoftBoxClasses(Classes):
    """A class is a box with smoothed (Gumbel) boundaries: centre and offset as for
    :class:`~mowl.representation.class_models.BoxClasses`, and the scale :math:`\\beta` of
    the smoothing, which the region carries.

    :param beta: Gumbel scale (``0``: hard boxes).
    :param offset: ``"abs"`` (default) or ``"softplus"``: how the stored offset is made
        non-negative.
    :param centres, offsets: Existing tables.

    Part (vocabulary value): ``soft box``. Used by: no method by default (``concept2box`` uses
    its contextual version).
    """

    region = SoftBox
    part = "soft box"
    exact_conjunction = True

    def __init__(self, size=None, embed_dim=None, beta=1.0, offset="abs", centres=None,
                 offsets=None):
        super().__init__()
        if offset not in ("abs", "softplus"):
            raise ValueError("offset must be 'abs' or 'softplus'.")
        self.beta = float(beta)
        self.offset_form = offset
        self.centres = centres if centres is not None else table(size, embed_dim)
        self.offsets = offsets if offsets is not None else table(len(self), self.embed_dim)

    def forward(self, ids):
        raw = self.offsets(ids)
        offset = th.abs(raw) if self.offset_form == "abs" else F.softplus(raw)
        return SoftBox(self.centres(ids), offset, self.beta)


class ContextualSoftBoxClasses(Classes):
    r"""A class is a soft box that depends on a context, a relation: the box of class
    :math:`c` in the context of relation :math:`r` is :math:`f_r(c)`, a two-layer network on
    the concatenation of the embeddings of the class and of the context, split into centre
    and offset (Concept2Box, Huang et al. 2023). Contexts are the ``num_relations`` relations,
    a context ``subclass_id`` for the subclass relation and a ``null_id`` for no context;
    :meth:`forward` gives the box in the null context.

    :param num_relations: Number of relations.
    :param beta: Gumbel scale.
    :param hidden_dim: Hidden size (default ``2 * embed_dim``).
    :param centres, contexts, project: Existing class table, context table (``num_relations
        + 2`` rows) and network.
    """

    region = SoftBox
    part = "soft box"
    exact_conjunction = True

    def __init__(self, size=None, embed_dim=None, num_relations=0, beta=1.0, hidden_dim=None,
                 centres=None, contexts=None, project=None):
        super().__init__()
        self.beta = float(beta)
        self.centres = centres if centres is not None else table(
            size, embed_dim, "normal", 0.5)
        dim = self.embed_dim
        self.contexts = contexts if contexts is not None else table(
            num_relations + 2, dim, "normal", 0.5)
        self.project = project if project is not None else nn.Sequential(
            nn.Linear(2 * dim, hidden_dim or 2 * dim), nn.ReLU(),
            nn.Linear(hidden_dim or 2 * dim, 2 * dim))
        self.num_relations = self.contexts.num_embeddings - 2
        self.subclass_id = self.num_relations
        self.null_id = self.num_relations + 1

    def in_context(self, ids, context):
        """The soft boxes of the classes ``ids`` in the context ``context`` (an int or
        ``LongTensor (B,)``: a relation, ``subclass_id`` or ``null_id``)."""
        if isinstance(context, int):
            context = th.full_like(ids, context)
        h = self.project(th.cat([self.centres(ids), self.contexts(context)], dim=-1))
        centre, raw = h.chunk(2, dim=-1)
        return SoftBox(centre, F.softplus(raw), self.beta, ids)

    def forward(self, ids):
        return self.in_context(ids, self.null_id)


# -- relation models ---------------------------------------------------------------------------

class ContextTransform(Relations):
    """The region of :math:`\\exists r.C` is the box of :math:`C` in the context of
    :math:`r`: a relation that is a function of the *class*, not only of its region. The
    region must therefore carry its class (``SoftBox.ids``). A relation has no image.

    :param classes: The :class:`ContextualSoftBoxClasses` that computes the boxes.
    """

    part = "neural network"

    def __init__(self, classes):
        super().__init__()
        self.classes = classes

    def __len__(self):
        return self.classes.num_relations

    def preimage(self, ids, region):
        if getattr(region, "ids", None) is None:
            raise ConstructorUnavailable(
                "ContextTransform needs the box of a named class (a region that carries "
                "its class), not a derived region.")
        return self.classes.in_context(region.ids, ids)


class RelationsByRegion(Relations):
    """A relation that is one thing for regions of one kind and another for regions of
    another kind (Concept2Box: a transform for the boxes of classes, a translation for the
    points of individuals). The first rule whose region class matches is used.

    :param rules: ``[(region class, relation model), ...]``.
    """

    def __init__(self, rules):
        super().__init__()
        self.rules = [(region, model) for region, model in rules]
        self.models = nn.ModuleList([model for _, model in self.rules])
        translations = [m for m in self.models if hasattr(m, "vectors")]
        #: The table of shift vectors, if one of the models has it.
        self.vectors = translations[0].vectors if translations else None

    @property
    def part(self):
        return self.models[0].part

    def __len__(self):
        return len(self.models[0])

    def _model(self, region):
        for cls, model in self.rules:
            if isinstance(region, cls):
                return model
        raise ConstructorNotImplemented(
            f"the relation model has no rule for {type(region).__name__} regions.")

    def image(self, ids, region):
        return self._model(region).image(ids, region)

    def preimage(self, ids, region):
        return self._model(region).preimage(ids, region)


register_class_model("point")(Points)
register_class_model("octagon product")(OctagonClasses)
register_class_model("triangle x box")(PrismClasses)
register_class_model("ellipsoid")(EllipsoidClasses)
register_class_model("soft box")(SoftBoxClasses)
