"""Class models: what a class is. A class model is a table of parameters and the rule that
turns a row into a region (:mod:`mowl.representation.regions`). The same classes serve as
individual models when individuals are small regions of the same kind.
"""

import inspect

import torch as th
import torch.nn as nn

from mowl.representation.regions import Ball, Box

__all__ = ["Classes", "BallClasses", "BoxClasses", "raw_tables"]


def _unit_rows(table):
    with th.no_grad():
        nn.init.uniform_(table.weight, a=-1, b=1)
        table.weight /= th.linalg.norm(table.weight, dim=1, keepdim=True)
    return table


def raw_tables(module):
    """The parameter tables of a class or relation model, by the name of the attribute that
    holds them: ``{"centres": Parameter (N, d), "radii": ...}`` (the weight of every
    embedding that is a direct child, and the parameters the model holds itself). They are
    the tensors the optimiser trains, not copies, for regimes that initialise, inspect or
    constrain the raw parameters (before any activation such as the absolute value)."""
    tables = {}
    for name, child in module.named_children():
        if isinstance(child, nn.Embedding):
            tables.setdefault(name, child.weight)
    for name, parameter in module.named_parameters(recurse=False):
        tables.setdefault(name, parameter)
    return tables


class Classes(nn.Module):
    """Base class of class models.

    :cvar region: The region class :meth:`forward` returns.
    :cvar part: The vocabulary entry (:class:`mowl.representation.ClassModel` value).
    :cvar exact_conjunction: Whether the regions are closed under intersection.
    :cvar conjunction_part: How the region of a conjunction is obtained, where the default
        (``"exact"`` or ``"approximate"``, from ``exact_conjunction``) does not say it
        (:class:`mowl.representation.Construction` value, for example ``"fuzzy operator"``).
    """

    region = None
    part = ""
    exact_conjunction = False
    conjunction_part = None

    @property
    def conjunction_kind(self):
        """``"exact"``, ``"approximate"`` or the construction the class model names."""
        if self.conjunction_part is not None:
            return self.conjunction_part
        return "exact" if self.exact_conjunction else "approximate"

    @property
    def extra_parts(self):
        """Parts of the geometry that only this class model can name (``{"negation": ...,
        "disjunction": ..., "family": ...}`` for fuzzy sets); added to
        :attr:`ComposedGeometry.parts <mowl.representation.ComposedGeometry.parts>`."""
        return {}

    def tables(self):
        """The raw parameter tables by attribute name (see :func:`raw_tables`): for a ball
        model ``{"centres": ..., "radii": ...}``, for boxes ``{"centres", "offsets"}``."""
        return raw_tables(self)

    def param_groups(self):
        """Parameter groups for an optimiser with manifolds (one group, no manifold)."""
        return [{"params": list(self.parameters()), "manifold": None}]

    def forward(self, ids):
        """The regions of the classes ``ids`` (``LongTensor (B,)``)."""
        raise NotImplementedError

    #: The context of :math:`C \\sqsubseteq D` for :meth:`in_context` (a class model with
    #: contexts sets it; for the others every context is the same).
    subclass_id = None

    def in_context(self, ids, context):
        """The regions of the classes ``ids`` in the context ``context`` (a relation, or
        :attr:`subclass_id`): the view of a class that depends on the triple it is read in
        (Concept2Box). A class model without contexts has one region per class, the same
        in every context."""
        return self(ids)

    def regions(self, ids, detach_size=False):
        """The regions of ``ids``; ``detach_size``: with their sizes detached
        (:meth:`Region.detach_size <mowl.representation.Region.detach_size>`), unless the
        class model's own ``forward`` takes the option (and detaches the parameter it stores
        the size in)."""
        if not detach_size:
            return self(ids)
        if "detach_size" in inspect.signature(self.forward).parameters:
            return self(ids, detach_size=True)
        return self(ids).detach_size()

    def __len__(self):
        return self.centres.num_embeddings

    @property
    def embed_dim(self):
        return self.centres.embedding_dim

    def all(self):
        """The regions of all classes."""
        return self(th.arange(len(self), device=self.centres.weight.device))


class BallClasses(Classes):
    """A class is a ball: a centre and a radius (stored signed, used through its absolute
    value).

    :param size: Number of classes.
    :param embed_dim: Dimension.
    :param centres, radii: Existing tables to use instead of new ones (the tables of
        another module, which then shares its parameters with this one).
    :param radii_init: ``"unit"`` (default): the stored radii start at :math:`\\pm 1`
        (uniform in :math:`[-1, 1]`, scaled to unit norm, as the centres); ``"uniform"``:
        uniform in :math:`[-1, 1]` (the individuals of ELEmbeddings; the radius is
        used through its absolute value, so none is negative).
    :param radius: How the radius is obtained from the stored parameter: ``"abs"``
        (default), ``"relu"`` (:math:`\\max(0, r)`, the EmEL++ Keras code) or ``"softplus"``
        (the training code of RegD's ``Elem``).

    Part (vocabulary value): ``ball``. Used by: ``elem``, ``emelpp``, ``emelvar``,
    ``regd_elem``.
    """

    region = Ball
    part = "ball"
    exact_conjunction = False

    def __init__(self, size=None, embed_dim=None, centres=None, radii=None, radius="abs",
                 radii_init="unit"):
        super().__init__()
        if radii_init not in ("unit", "uniform"):
            raise ValueError("radii_init must be 'unit' or 'uniform'.")
        if radius not in ("abs", "relu", "softplus"):
            raise ValueError("radius must be 'abs', 'relu' or 'softplus'.")
        self.radius_form = radius
        self.centres = centres if centres is not None else _unit_rows(
            nn.Embedding(size, embed_dim))
        if radii is None:
            radii = nn.Embedding(len(self), 1)
            if radii_init == "unit":
                _unit_rows(radii)
            else:
                with th.no_grad():
                    nn.init.uniform_(radii.weight, a=-1, b=1)
        self.radii = radii

    def forward(self, ids):
        raw = self.radii(ids)
        radius = {"abs": th.abs, "relu": th.relu,
                  "softplus": nn.functional.softplus}[self.radius_form](raw)
        return Ball(self.centres(ids), radius, raw)


class BoxClasses(Classes):
    """A class is an axis-parallel box: a centre and half-widths (stored signed, used
    through their absolute value).

    :param centres, offsets: Existing tables to use instead of new ones.
    :param init: ``"separate"`` (default): centres and offsets are each drawn from
        :math:`U(-1, 1)^n` and scaled to unit norm. ``"joint"`` (the Box2EL and TransBox
        codes): the centre and the offset of a class form one row of :math:`U(-1, 1)^{2n}`
        scaled to unit norm. (Only for new tables.)

    Part (vocabulary value): ``box``. Used by: ``elbe``, ``regd``, ``transbox``.
    """

    region = Box
    part = "box"
    exact_conjunction = True

    def __init__(self, size=None, embed_dim=None, centres=None, offsets=None,
                 init="separate"):
        super().__init__()
        if init not in ("separate", "joint"):
            raise ValueError("init must be 'separate' or 'joint'.")
        self.init = init
        self.centres = centres if centres is not None else _unit_rows(
            nn.Embedding(size, embed_dim))
        self.offsets = offsets if offsets is not None else _unit_rows(
            nn.Embedding(len(self), self.embed_dim))
        if init == "joint" and centres is None and offsets is None:
            # as the codes do it: one row of U(-1, 1)^(2n), scaled to unit norm as a whole
            weight = th.empty(len(self), 2 * self.embed_dim)
            nn.init.uniform_(weight, a=-1, b=1)
            weight = weight / th.linalg.norm(weight, dim=1, keepdim=True)
            with th.no_grad():
                self.centres.weight.copy_(weight[:, :self.embed_dim])
                self.offsets.weight.copy_(weight[:, self.embed_dim:])

    def forward(self, ids):
        return Box(self.centres(ids), th.abs(self.offsets(ids)))
