"""Class models of the methods that embed a hierarchy (group ``hierarchy``).

=========================  =====================================  ==========================
class model                what a class is                        used by
=========================  =====================================  ==========================
:class:`UnitCubeBoxes`     a box in the unit cube                 box lattice, Gumbel boxes
:class:`CodedBoxes`        a box with binary codes                BCBox
:class:`OrderPoints`       a vector of an ordered vector space    order embeddings, CatE
:class:`PointClasses`      a point of a manifold                  shadow cones, HiT, OnT
:class:`ConePoints`        a point with its geodesic cone         entailment cones
:class:`DiskClasses`       a point with a (signed) radius         disk embeddings
:class:`TextPoints`        a point computed from a text           HiT, OnT
=========================  =====================================  ==========================

Every class model that has a table of its own names it ``centres`` (so that a
:class:`~mowl.representation.ComposedGeometry` can use it) and can be built around the
table of an existing module (``table=``), which then shares its parameters with it.
"""

import torch as th
import torch.nn as nn

from mowl.nn.order import cones as C
from mowl.nn.order.manifolds import (Euclidean, Hyperboloid, PoincareBall, Sphere,
                                     UpperHalfSpace)
from mowl.representation.class_models import Classes
from mowl.representation.library.hierarchy_regions import (
    CodedBox, ConePoint, CornerBox, CurvedBall, Disk, OrderCone, Point)
from mowl.representation.methods import register_class_model

__all__ = ["HierarchyClasses", "UnitCubeBoxes", "CodedBoxes", "OrderPoints", "PointClasses",
           "ConePoints", "DiskClasses", "TextPoints", "init_points"]


class HierarchyClasses(Classes):
    """Base of the class models of this group: a size and a device that do not depend on a
    table called ``centres`` (the encoder of a text model has none)."""

    size = 0
    _embed_dim = 0
    #: There is no conjunction unless the regions are closed under it (boxes).
    conjunction_part = "none"

    def __len__(self):
        return self.size

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def device(self):
        for p in self.parameters():
            return p.device
        return th.device("cpu")

    def all(self):
        return self(th.arange(len(self), device=self.device))


# -- boxes in the unit cube --------------------------------------------------------------
class UnitCubeBoxes(HierarchyClasses):
    r"""A class is a box in :math:`[0, 1]^d`, stored as (minimum, :math:`\Delta` = maximum
    minus minimum) in one table ``centres`` of ``2 d`` columns (Vilnis et al. 2018,
    Section 4.3). The cube is a constraint of this class model: :meth:`project` is the
    Euclidean projection that the geometry requires after every optimiser step (minimum in
    :math:`[0, 1 - \Delta_{\min}]`, :math:`\Delta \in [\Delta_{\min}, 1 - \min]`); a
    method that trains the geometry has to call it (the models of mOWL do, through
    ``module.project``).

    :param min_delta: :math:`\Delta_{\min}`.
    :param init_side: Sides are initialised uniformly in this range.
    :param table: An existing table (``(size, 2 d)``) to use instead of a new one.
    :param soft: Only the label (:attr:`part`): ``"soft box"`` for the models that read the
        corners as locations of Gumbel variables. The class model is the same.
    """

    region = CornerBox
    part = "box"
    exact_conjunction = True
    conjunction_part = None

    def __init__(self, size=None, embed_dim=None, min_delta=0.01, init_side=(0.3, 0.9),
                 table=None, soft=False):
        super().__init__()
        self.min_delta = float(min_delta)
        if soft:
            self.part = "soft box"
        if table is None:
            table = nn.Embedding(size, 2 * embed_dim)
            with th.no_grad():
                side = th.empty(size, embed_dim).uniform_(*init_side)
                low = th.rand(size, embed_dim) * (1.0 - side)
                table.weight.copy_(th.cat([low, side], dim=1))
        self.centres = table
        self.size = table.num_embeddings
        self._embed_dim = table.embedding_dim // 2

    def forward(self, ids, detach_size=False):
        """The boxes of ``ids``; with ``detach_size`` the sizes carry no gradient (the box
        lattice model does not update :math:`\\Delta` through negative pairs)."""
        p = self.centres(ids)
        low, delta = p[:, :self._embed_dim], p[:, self._embed_dim:]
        if detach_size:
            delta = delta.detach()
        low = low.clamp(0.0, 1.0)
        delta = th.maximum(delta, th.full_like(delta, self.min_delta))
        return CornerBox.from_corners(low, low + delta)

    @th.no_grad()
    def project(self):
        """Project the parameters back onto the cube."""
        w = self.centres.weight
        d = self._embed_dim
        low = w[:, :d].clamp(0.0, 1.0 - self.min_delta)
        delta = th.minimum(w[:, d:].clamp(min=self.min_delta), 1.0 - low)
        w.copy_(th.cat([low, delta], dim=1))


# -- boxes with binary codes -------------------------------------------------------------
class CodedBoxes(HierarchyClasses):
    r"""A box :math:`\prod_i [x^-_i, x^+_i]` with unconstrained corners and, as the variant
    says, no code, one code per class or global code words (Zhang et al. 2022):

    * ``"gbox"``: no code;
    * ``"vbc"``: a code :math:`u^\diamond \in [0, 1]^d` per class (sigmoid of logits);
    * ``"gbc"``: ``num_codes`` code words :math:`B_{\cdot j} \in [0, 1]^d`, shared by all
      classes.

    Only the last ``code_dims`` dimensions have trainable codes, the others are fixed to 1.
    The class model stores the codes; reading them is the business of the measure.

    :param tables: A module (:class:`~mowl.nn.graph.bcbox.BinaryCodeBoxModule`) whose
        parameters ``box_min``, ``box_max`` and ``code_logits`` are used instead of new ones.
    """

    region = CodedBox
    part = "soft box"
    exact_conjunction = True
    conjunction_part = None

    def __init__(self, size=None, embed_dim=None, variant="vbc", num_codes=2, code_dims=None,
                 code_init=0.0, tables=None):
        super().__init__()
        if variant not in ("gbox", "gbc", "vbc"):
            raise ValueError("variant must be 'gbox', 'gbc' or 'vbc'.")
        self.variant = variant
        if tables is not None:
            self.box_min, self.box_max = tables.box_min, tables.box_max
            self.size, self._embed_dim = tables.box_min.shape
            self.code_dims = tables.code_dims
            self.num_codes = tables.num_codes
            self.code_logits = tables.code_logits
            return
        if code_dims is None:
            code_dims = embed_dim
        self.size, self._embed_dim = size, embed_dim
        self.code_dims = code_dims if variant != "gbox" else 0
        self.num_codes = num_codes if variant == "gbc" else 0
        corners = th.sort(th.randn(size, embed_dim, 2), dim=-1).values
        self.box_min = nn.Parameter(corners[..., 0].contiguous())
        self.box_max = nn.Parameter(corners[..., 1].contiguous())
        if variant == "gbc":
            self.code_logits = nn.Parameter(th.randn(self.code_dims, num_codes) + code_init)
        elif variant == "vbc":
            self.code_logits = nn.Parameter(th.randn(size, self.code_dims) + code_init)
        else:
            self.register_parameter("code_logits", None)

    def global_codes(self):
        """:math:`B \\in [0, 1]^{d \\times k}` (``"gbc"``)."""
        fixed = th.ones(self._embed_dim - self.code_dims, self.num_codes,
                        device=self.box_min.device, dtype=self.box_min.dtype)
        return th.cat((fixed, th.sigmoid(self.code_logits)), dim=0)

    def vertex_codes(self, ids):
        """:math:`u^\\diamond \\in [0, 1]^d` of the classes ``ids`` (``"vbc"``)."""
        learned = th.sigmoid(self.code_logits[ids])
        fixed = th.ones(*ids.shape, self._embed_dim - self.code_dims,
                        device=learned.device, dtype=learned.dtype)
        return th.cat((fixed, learned), dim=-1)

    def forward(self, ids):
        code = self.vertex_codes(ids) if self.variant == "vbc" else None
        global_code = self.global_codes() if self.variant == "gbc" else None
        return CodedBox.from_corners(self.box_min[ids], self.box_max[ids], code=code,
                                     global_code=global_code)

    def code_regularization(self):
        """:math:`\\lVert 1 - B \\rVert_1` of the trainable codes (0 for ``"gbox"``)."""
        if self.code_logits is None:
            return th.zeros((), device=self.box_min.device)
        return (1.0 - th.sigmoid(self.code_logits)).abs().sum()


# -- points ------------------------------------------------------------------------------
def init_points(manifold, n, dim, scale=0.1, max_norm=0.5, generator=None):
    """New points of ``manifold`` (``dim`` is the dimension of the space; points of the
    hyperboloid have ``dim + 1`` coordinates):

    * Poincare ball: a random direction, norm uniform between the smallest norm the ball
      allows and ``max_norm`` (the initialisation of the entailment cones);
    * half-space: uniform in :math:`[-s, s]^n` around :math:`(0, \\dots, 0, 1)` (Nickel and
      Kiela);
    * hyperboloid: uniform in :math:`[-s, s]^n`, lifted to the hyperboloid;
    * sphere: uniform in :math:`[-s, s]^n` with a little noise, normalised;
    * Euclidean: uniform in :math:`[-s, s]^n`.
    """
    def uniform():
        return (th.rand(n, dim, generator=generator) * 2 - 1) * scale
    if isinstance(manifold, PoincareBall):
        d = th.randn(n, dim, generator=generator)
        d = d / d.norm(dim=-1, keepdim=True)
        low = manifold.min_norm
        high = max(max_norm, low * 1.5)
        return d * (low + (high - low) * th.rand(n, 1, generator=generator))
    if isinstance(manifold, UpperHalfSpace):
        x = uniform()
        x[:, -1] += 1.0
        return manifold.project(x)
    if isinstance(manifold, Hyperboloid):
        return Hyperboloid.from_spatial(uniform())
    if isinstance(manifold, Sphere):
        return manifold.project(uniform() + 1e-3 * th.randn(n, dim, generator=generator))
    if isinstance(manifold, Euclidean):
        return uniform()
    raise TypeError(f"No initialisation for {type(manifold).__name__}.")


class PointClasses(HierarchyClasses):
    """A class is a point of a manifold (a table ``centres``; the coordinates are those of
    the model of the manifold, so a point of the hyperboloid has ``embed_dim + 1``).

    :param manifold: A manifold of :mod:`mowl.nn.order.manifolds` (default: the Poincare
        ball).
    :param init_scale, init_max_norm: See :func:`init_points`.
    :param table: An existing table to use instead of a new one.
    :param seed: Seed of the initialisation (new tables only).
    """

    region = Point
    part = "point"

    def __init__(self, size=None, embed_dim=None, manifold=None, init_scale=0.1,
                 init_max_norm=0.5, table=None, seed=None):
        super().__init__()
        self.manifold = manifold if manifold is not None else PoincareBall()
        if table is None:
            gen = None if seed is None else th.Generator().manual_seed(seed)
            table = nn.Embedding(size, embed_dim + isinstance(self.manifold, Hyperboloid))
            with th.no_grad():
                table.weight.copy_(init_points(self.manifold, size, embed_dim, init_scale,
                                               init_max_norm, gen))
        self.centres = table
        self.size = table.num_embeddings
        self._embed_dim = table.embedding_dim - isinstance(self.manifold, Hyperboloid)

    def forward(self, ids):
        return Point(self.centres(ids), self.manifold, ids)

    def param_groups(self):
        return [{"params": [self.centres.weight], "manifold": self.manifold}]


class ConePoints(PointClasses):
    r"""A point :math:`x` of the Poincare ball with the entailment cone of half-aperture
    :math:`\psi(x) = \arcsin(K (1 - \lVert x\rVert^2) / \lVert x\rVert)` at it (Ganea et al.
    2018, Eq. 26); the points keep out of the ball of radius :math:`\epsilon(K)` around the
    origin, where the cone is not defined.

    :param K: The constant :math:`K` of the cone (0.1 in the paper).
    """

    region = ConePoint
    part = "geodesic cone"

    def __init__(self, size=None, embed_dim=None, K=0.1, init_max_norm=0.5, table=None,
                 seed=None, manifold=None):
        self.K = float(K)
        self.epsilon = C.entailment_cone_min_norm(self.K)
        if manifold is None:
            manifold = PoincareBall(min_norm=self.epsilon, max_norm=1 - 1e-5)
        super().__init__(size, embed_dim, manifold, init_max_norm=init_max_norm, table=table,
                         seed=seed)

    def forward(self, ids):
        x = self.centres(ids)
        return ConePoint(x, self.manifold, ids, C.cone_half_aperture(x, self.K))


class DiskClasses(PointClasses):
    """A generalised formal disk: a point of a manifold and a real radius (a table
    ``radii``, initialised to 0). Default manifold: the hyperboloid.

    :param radii: An existing table of radii to use instead of a new one.
    """

    region = Disk
    part = "disk"

    def __init__(self, size=None, embed_dim=None, manifold=None, init_scale=0.1, table=None,
                 radii=None, seed=None):
        super().__init__(size, embed_dim, manifold if manifold is not None else Hyperboloid(),
                         init_scale=init_scale, table=table, seed=seed)
        if radii is None:
            radii = nn.Embedding(self.size, 1)
            nn.init.zeros_(radii.weight)
        self.radii = radii

    def forward(self, ids):
        return Disk(self.centres(ids), self.manifold, ids, self.radii(ids)[:, 0])

    def param_groups(self):
        return [{"params": [self.centres.weight], "manifold": self.manifold},
                {"params": [self.radii.weight], "manifold": None}]


# -- vectors of an ordered space ---------------------------------------------------------
class OrderPoints(HierarchyClasses):
    r"""A class is a vector of :math:`\mathbb{R}^d` ordered componentwise; the order is read
    by :class:`~.hierarchy_measures.OrderViolation`.

    :param nonnegative: Use the absolute value of the parameters, so that the vectors lie in
        :math:`\mathbb{R}^d_+` (order embeddings, Vendrov et al. 2016, Eq. 6); ``False`` for
        free vectors (CatE).
    :param init: ``"uniform"`` (in ``[0, init_scale]``) or ``"xavier"``.
    :param table: An existing table to use instead of a new one.
    """

    region = OrderCone
    part = "order cone"

    def __init__(self, size=None, embed_dim=None, nonnegative=True, init="uniform",
                 init_scale=1.0, table=None):
        super().__init__()
        if init not in ("uniform", "xavier"):
            raise ValueError("init must be 'uniform' or 'xavier'.")
        self.nonnegative = bool(nonnegative)
        if table is None:
            table = nn.Embedding(size, embed_dim)
            if init == "uniform":
                nn.init.uniform_(table.weight, a=0.0, b=init_scale)
            else:
                nn.init.xavier_uniform_(table.weight)
        self.centres = table
        self.size = table.num_embeddings
        self._embed_dim = table.embedding_dim
        self.manifold = Euclidean()

    def forward(self, ids):
        x = self.centres(ids)
        return OrderCone(x.abs() if self.nonnegative else x, self.manifold, ids)


# -- points computed from texts ----------------------------------------------------------
class TextPoints(HierarchyClasses):
    """A class is the point that a text encoder computes from its verbalisation (HiT, OnT).
    There is no table: the parameters are those of the encoder.

    :param encoder: A module mapping a list of strings to a ``(n, d)`` tensor
        (:mod:`mowl.nn.lm.encoders`).
    :param texts: The verbalisation of every class id, to look classes up by id. Without it,
        the "ids" of :meth:`forward` are lists of strings, which also allows classes that were
        not seen in training (the point of a text model).
    :param manifold: The ball the outputs are read in; default: curvature :math:`1/d`, the
        ball that circumscribes the hypercube :math:`[-1, 1]^d` (HiT, Section 3).
    """

    region = Point
    part = "point"

    def __init__(self, encoder, texts=None, manifold=None, curvature=None):
        super().__init__()
        self.encoder = encoder
        self.texts = None if texts is None else list(texts)
        self.size = 0 if texts is None else len(self.texts)
        self._embed_dim = int(encoder.dim)
        if manifold is None:
            manifold = CurvedBall(curvature if curvature is not None else 1.0 / self._embed_dim)
        self.manifold = manifold

    def forward(self, ids):
        if isinstance(ids, (list, tuple)) and all(isinstance(t, str) for t in ids):
            texts = list(ids)
        elif self.texts is None:
            raise TypeError("This class model has no list of texts: pass a list of strings.")
        else:
            texts = [self.texts[int(i)] for i in th.as_tensor(ids).reshape(-1)]
        return Point(self.encoder(texts), self.manifold)


# -- registration ------------------------------------------------------------------------
@register_class_model("unit box")
def _unit_box(size, embed_dim):
    return UnitCubeBoxes(size, embed_dim)


@register_class_model("order cone")
def _order_cone(size, embed_dim):
    return OrderPoints(size, embed_dim)


@register_class_model("manifold point")
def _point(size, embed_dim):
    return PointClasses(size, embed_dim)


@register_class_model("geodesic cone")
def _geodesic_cone(size, embed_dim):
    return ConePoints(size, embed_dim)


@register_class_model("disk")
def _disk(size, embed_dim):
    return DiskClasses(size, embed_dim)
