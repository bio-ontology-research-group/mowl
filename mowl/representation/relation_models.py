"""Relation models: what a relation is. A relation model maps regions to regions:
:meth:`~Relations.image` gives the things related *from* a region,
:meth:`~Relations.preimage` the things related *to* it, which is the region of
:math:`\\exists r.C`.

Because only the relation model knows what a relation is, it also answers the axioms about
relations, through one interface: :meth:`~Relations.role_inclusion`
(:math:`r \\sqsubseteq s`) and :meth:`~Relations.role_chain`
(:math:`r \\circ t \\sqsubseteq s`). Both take the measure and the margin of the objective,
which a relation model that compares regions (boxes of translations) uses and one that
compares parameters (shift vectors) may ignore. A relation model that defines them lets a
:class:`~mowl.representation.ComposedGeometry` train on the EL++ role axioms.
"""

import torch as th
import torch.nn as nn

from mowl.representation.operations import Operations, generic
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["Relations", "Translation"]


class Relations(Operations, nn.Module):
    """Base class of relation models.

    :cvar part: The vocabulary entry (:class:`mowl.representation.RelationModel` value).

    The optional operations of :attr:`OPERATIONS` (:mod:`mowl.representation.operations`)
    have generic definitions here in terms of :meth:`preimage` and :meth:`image`; a relation
    model overrides them where its own reading differs (head and tail boxes, a box of
    translations, a relation with a range).
    """

    OPERATIONS = {
        "subsumed_negative": "the negative loss of C subClassOf r some D of Box2EL and DELE, "
                             "(delta - mu(C, exists r.D))^2 + (delta - mu(D, image of C))^2 "
                             "with mu the measure's meets",
        "preimage_all": "the region of the points every successor of which lies in C "
                        "(exists r^all.C of TransBox); generic for a functional relation: "
                        "the preimage",
        "range": "the region of the points that are successors of some point; generic for "
                 "a relation onto the whole space: None (no constraint)",
        "image_with_error": "the image of a region with the error of the relation (ELK-Em); "
                            "generic: the image",
    }

    part = ""
    #: The region classes whose regions :meth:`image` and :meth:`preimage` map (``None``:
    #: every region that can be translated or whatever the model needs; a model that maps
    #: only some families says which, so that a combination that cannot read
    #: :math:`\\exists r.C` raises :class:`ConstructorUnavailable` instead of failing inside
    #: a formula).
    accepts = None
    #: Whether a relation is a function (every point has exactly one successor, given by
    #: :meth:`successor`); the preimage of a region is then the set of points whose
    #: successor lies in it.
    functional = False
    #: Whether every point is the successor of some point (the range is the whole space).
    surjective = False

    #: The maps that are applied to every component of a region made of components (a
    #: union of boxes) when the relation model maps the components but not the union: the
    #: image and the preimage of a union are the unions of the images and preimages.
    LIFTED = ("image", "preimage", "sharp_preimage", "image_with_error")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in cls.LIFTED:
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "lifted", False):
                setattr(cls, name, _lifted(method))

    def maps(self, region_class):
        """Whether the relation model maps the regions of ``region_class``, itself or (for a
        region made of components) component by component."""
        if self.accepts is None or issubclass(region_class, self.accepts):
            return True
        component = getattr(region_class, "component", None)
        return component is not None and issubclass(component, self.accepts)

    def check_region(self, region_class):
        """Raise :class:`ConstructorUnavailable` if the regions of ``region_class`` are not
        mapped by this relation model."""
        if not self.maps(region_class):
            raise ConstructorUnavailable(
                f"{type(self).__name__} maps {[c.__name__ for c in self.accepts]} regions, "
                f"not {region_class.__name__}: this combination of parts cannot read "
                f"exists r.C.")

    def related(self, ids, x, y):
        """``(B, P)`` boolean: whether the points ``x[p]`` and ``y[p]`` (``(P, d)`` each) are
        a pair of the relation of row ``b`` of ``ids``. This is the meaning of the relation
        model: :meth:`preimage` must be the set of points with a successor in the region,
        :meth:`image` the set of successors of its points, and the role axioms must hold
        for these pairs. A relation model that is not a set of pairs of points (a relation
        between regions, a ranking score) has no such test."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} does not say which pairs of points it relates.")

    def image(self, ids, region):
        """:math:`\\{y : (x, y) \\in r, x \\in \\text{region}\\}` for the relations
        ``ids``."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no image.")

    def preimage(self, ids, region):
        """:math:`\\{x : (x, y) \\in r, y \\in \\text{region}\\}`: the region of
        :math:`\\exists r.C`."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no preimage.")

    def tables(self):
        """The raw parameter tables by attribute name (``{"vectors": ...}`` for a
        translation; see :func:`~mowl.representation.class_models.raw_tables`)."""
        from mowl.representation.class_models import raw_tables
        return raw_tables(self)

    def sharp_preimage(self, ids, region):
        """The preimage without any tolerance of the relation (what a *negative* axiom is
        read against). Default: the preimage."""
        return self.preimage(ids, region)

    def tolerance(self, ids):
        """How much the relation gives up of the exact image, per relation (``(B, 1)`` or
        ``0.0``); an objective may charge it (``tolerance_weight``). Default: none."""
        return 0.0

    def subsumed(self, measure, sub, ids, filler, margin=0.0):
        """Cost of :math:`C \\sqsubseteq \\exists r.D` for the region ``sub`` of
        :math:`C` and the region ``filler`` of :math:`D`. Default: ``sub`` lies inside the
        preimage of ``filler``. A relation model whose :math:`\\exists r.D` needs a second
        condition (head and tail boxes) defines its own reading."""
        return measure.inside(sub, self.preimage(ids, filler), margin)

    @generic
    def subsumed_negative(self, measure, sub, ids, filler, margin, delta):
        """The negative loss of :math:`C \\sqsubseteq \\exists r.D` of Box2EL and DELE:
        :math:`(\\delta - \\mu(C, \\exists r.D))^2 + (\\delta - \\mu(D, r(C)))^2` with
        :math:`\\mu` the measure's ``meets`` (with ``margin``): the two pairs are pushed
        :math:`\\delta` apart."""
        first = measure.meets(sub, self.sharp_preimage(ids, filler), margin)
        second = measure.meets(filler, self.image(ids, sub), margin)
        return (delta - first).square() + (delta - second).square()

    @generic
    def preimage_all(self, ids, region, clamp=True):
        """:math:`\\exists r^{all}.C`, the points all successors of which lie in ``region``.
        For a functional relation (one successor per point) this is the preimage; other
        relation models define it or refuse."""
        if not self.functional:
            raise ConstructorUnavailable(self._missing("preimage_all") + " It has a "
                                         "generic definition only for functional relations.")
        return self.preimage(ids, region)

    @generic
    def range(self, ids):
        """The region of the successors of the relations ``ids``; ``None`` for a relation
        onto the whole space (no constraint), which is all a generic definition can say."""
        if not self.surjective:
            raise ConstructorUnavailable(self._missing("range") + " It has a generic "
                                         "definition only for relations onto the space.")
        return None

    @generic
    def image_with_error(self, ids, region):
        """The image of ``region`` with the error of the relation; a relation without one:
        the image."""
        return self.image(ids, region)

    def role_inclusion(self, sub, sup, measure, margin=0.0):
        """Cost ``(B, 1)`` of :math:`r \\sqsubseteq s` for the relation ids ``sub``,
        ``sup``; ``measure`` and ``margin`` are those of the objective."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} does not define role_inclusion(...), so role "
            f"inclusions cannot be read.")

    def role_chain(self, first, second, sup, measure, margin=0.0):
        """Cost ``(B, 1)`` of :math:`r_1 \\circ r_2 \\sqsubseteq s` (first :math:`r_1`, then
        :math:`r_2`)."""
        raise ConstructorUnavailable(
            f"{type(self).__name__} does not define role_chain(...), so role chains "
            f"cannot be read.")

    @property
    def reads_role_axioms(self):
        """Whether the relation model defines both role axioms."""
        cls = type(self)
        return (cls.role_inclusion is not Relations.role_inclusion
                and cls.role_chain is not Relations.role_chain)

    def __len__(self):
        return self.vectors.num_embeddings


def _lifted(method):
    """``method(ids, region, ...)`` applied to every component of ``region`` when the relation
    model maps its components and not the region itself (exact: a relation maps a union to
    the union of the maps of its parts)."""
    import functools

    @functools.wraps(method)
    def lifted(self, ids, region, *args, **kwargs):
        accepts = self.accepts
        component = getattr(type(region), "component", None)
        if accepts is not None and not isinstance(region, accepts) \
                and component is not None and issubclass(component, accepts):
            flat, count = region.flat_components()
            mapped = method(self, ids.repeat_interleave(count), flat, *args, **kwargs)
            return region.from_components(mapped)
        return method(self, ids, region, *args, **kwargs)
    lifted.lifted = True
    return lifted


def _direction_term(a, b, direction):
    """:math:`|1 - \\cos(a, b)|` per row (``"row"``), or the term of the EmEL++ Keras code
    (``"batch"``): every coordinate is normalised over the *batch* (``tf.nn.l2_normalize(x,
    0)``) and all products are summed, giving one number per batch that is added to every
    row."""
    if direction == "batch":
        total = (th.nn.functional.normalize(a, dim=0)
                 * th.nn.functional.normalize(b, dim=0)).sum()
        return th.abs(1 - total).expand(len(a), 1)
    return th.abs(1 - th.nn.functional.cosine_similarity(a, b, dim=1).unsqueeze(1))


class Translation(Relations):
    """A relation is one shift vector: :math:`(x, y) \\in r` iff :math:`y = x + v_r`
    (TransE; ELEmbeddings, ELBE and many others). A function: every point has exactly one
    successor, so one-to-many relations cannot be represented.

    Role axioms: :math:`r \\sqsubseteq s` holds iff :math:`v_r = v_s` and
    :math:`r_1 \\circ r_2 \\sqsubseteq s` iff :math:`v_{r_1} + v_{r_2} = v_s`. The cost is the
    distance of the two sides; equality has no strict margin, so only a slack (a negative
    margin) is read.

    :param vectors: An existing table to use instead of a new one.
    :param direction: ``None`` (default): the cost of a role axiom is the distance of the two
        sides. ``"row"``: the term :math:`|1 - \\cos(a, b)|` of EmEL++ (Mondal et al. 2021,
        Eq. 8-9) is added per row; ``"batch"``: the version of the authors' Keras code,
        normalised over the batch, one number added to every row.

    Part (vocabulary value): ``translation``. Used by: ``eike``, ``elbe``, ``elem``, ``emelpp``,
    ``multielbe``, ``octagon``, ``prism``, ``prism(reading=structural)``, ``regd``,
    ``regd_elem``.
    """

    part = "translation"
    functional = True
    surjective = True

    def __init__(self, size=None, embed_dim=None, vectors=None, direction=None):
        super().__init__()
        if direction not in (None, "row", "batch"):
            raise ValueError("direction must be None, 'row' or 'batch'.")
        self.direction = direction
        if vectors is None:
            vectors = nn.Embedding(size, embed_dim)
            with th.no_grad():
                nn.init.uniform_(vectors.weight, a=-1, b=1)
                vectors.weight /= th.linalg.norm(vectors.weight, dim=1, keepdim=True)
        self.vectors = vectors

    def successor(self, ids, x):
        """The successor :math:`x + v_r` of every point ``x`` (``(P, d)``) for every relation
        of ``ids``: ``(B, P, d)``."""
        return x.unsqueeze(0) + self.vectors(ids).unsqueeze(1)

    def related(self, ids, x, y, tolerance=1e-5):
        """:math:`y = x + v_r` (up to ``tolerance``, a translation relates sets of measure
        zero)."""
        gap = th.linalg.norm(y.unsqueeze(0) - self.successor(ids, x), dim=-1)
        return gap <= tolerance

    def image(self, ids, region):
        return region.translate(self.vectors(ids))

    def preimage(self, ids, region):
        return region.translate(-self.vectors(ids))

    def _role_cost(self, left, right, margin):
        cost = th.relu(th.linalg.norm(right - left, dim=1, keepdim=True) + min(margin, 0.0))
        if self.direction is not None:
            cost = cost + _direction_term(left, right, self.direction)
        return cost

    def role_inclusion(self, sub, sup, measure=None, margin=0.0):
        return self._role_cost(self.vectors(sub), self.vectors(sup), margin)

    def role_chain(self, first, second, sup, measure=None, margin=0.0):
        return self._role_cost(self.vectors(first) + self.vectors(second), self.vectors(sup),
                               margin)
